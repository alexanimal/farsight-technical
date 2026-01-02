# AWS ECS Deployment Guide

This document outlines how to deploy the Farsight Technical server application to AWS ECS (Elastic Container Service).

## Architecture Overview

The deployment consists of several components:

1. **API Server Container** - FastAPI application serving HTTP requests
2. **Temporal Worker Container** - Executes Temporal workflows and activities
3. **Temporal Server** - Workflow orchestration engine (can be deployed separately or as a container)
4. **PostgreSQL** - Database for Temporal and application data (AWS RDS)
5. **Redis** - Cache and conversation history (AWS ElastiCache)

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed locally
- ECR repository created for container images
- VPC with public and private subnets configured
- Security groups configured for services

## Step 1: Build and Push Docker Images

### 1.1 Create ECR Repository

```bash
aws ecr create-repository --repository-name farsight-server --region us-east-1
```

### 1.2 Authenticate Docker to ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

### 1.3 Build and Tag Images

```bash
# Build the server image
cd server
docker build -t farsight-server:latest .

# Tag for ECR
docker tag farsight-server:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/farsight-server:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/farsight-server:latest
```

## Step 2: Set Up AWS Infrastructure

### 2.1 RDS PostgreSQL Database

Create an RDS PostgreSQL instance for Temporal and application data:

```bash
aws rds create-db-instance \
  --db-instance-identifier farsight-postgres \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 14.9 \
  --master-username postgres \
  --master-user-password <secure-password> \
  --allocated-storage 100 \
  --vpc-security-group-ids <security-group-id> \
  --db-subnet-group-name <subnet-group-name> \
  --backup-retention-period 7 \
  --multi-az
```

**Note:** Store database credentials in AWS Secrets Manager for secure access.

### 2.2 ElastiCache Redis Cluster

Create an ElastiCache Redis cluster for conversation history:

```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id farsight-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --vpc-security-group-ids <security-group-id> \
  --subnet-group-name <subnet-group-name>
```

### 2.3 Temporal Server (Optional)

You can deploy Temporal server as a separate ECS service or use Temporal Cloud. For self-hosted:

- Deploy Temporal server container in ECS
- Use the same RDS PostgreSQL instance (separate database)
- Configure Temporal UI as a separate service

## Step 3: Configure Secrets in AWS Secrets Manager

Store sensitive environment variables in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name farsight/server-secrets \
  --secret-string '{
    "OPENAI_API_KEY": "your-openai-key",
    "PINECONE_API_KEY": "your-pinecone-key",
    "POSTGRES_PASSWORD": "your-postgres-password",
    "REDIS_PASSWORD": "your-redis-password",
    "API_KEY": "your-api-key"
  }'
```

## Step 4: Create ECS Task Definitions

### 4.1 API Server Task Definition

Create `task-definition-api.json`:

```json
{
  "family": "farsight-api-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "api-server",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/farsight-server:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "POSTGRES_HOST",
          "value": "<rds-endpoint>"
        },
        {
          "name": "POSTGRES_PORT",
          "value": "5432"
        },
        {
          "name": "POSTGRES_DB_NAME",
          "value": "farsight"
        },
        {
          "name": "POSTGRES_USER",
          "value": "postgres"
        },
        {
          "name": "REDIS_HOST",
          "value": "<elasticache-endpoint>"
        },
        {
          "name": "REDIS_PORT",
          "value": "6379"
        },
        {
          "name": "TEMPORAL_ADDRESS",
          "value": "<temporal-endpoint>:7233"
        },
        {
          "name": "TEMPORAL_NAMESPACE",
          "value": "default"
        },
        {
          "name": "TEMPORAL_TASK_QUEUE",
          "value": "orchestrator-task-queue"
        },
        {
          "name": "PINECONE_INDEX",
          "value": "default-index"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:OPENAI_API_KEY::"
        },
        {
          "name": "PINECONE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:PINECONE_API_KEY::"
        },
        {
          "name": "POSTGRES_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:POSTGRES_PASSWORD::"
        },
        {
          "name": "REDIS_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:REDIS_PASSWORD::"
        },
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:API_KEY::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/farsight-api-server",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"
        ],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

Register the task definition:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition-api.json
```

### 4.2 Temporal Worker Task Definition

Create `task-definition-worker.json`:

```json
{
  "family": "farsight-worker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "temporal-worker",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/farsight-server:latest",
      "essential": true,
      "command": ["python", "-m", "src.temporal.worker"],
      "environment": [
        {
          "name": "POSTGRES_HOST",
          "value": "<rds-endpoint>"
        },
        {
          "name": "POSTGRES_PORT",
          "value": "5432"
        },
        {
          "name": "POSTGRES_DB_NAME",
          "value": "farsight"
        },
        {
          "name": "POSTGRES_USER",
          "value": "postgres"
        },
        {
          "name": "REDIS_HOST",
          "value": "<elasticache-endpoint>"
        },
        {
          "name": "REDIS_PORT",
          "value": "6379"
        },
        {
          "name": "TEMPORAL_ADDRESS",
          "value": "<temporal-endpoint>:7233"
        },
        {
          "name": "TEMPORAL_NAMESPACE",
          "value": "default"
        },
        {
          "name": "TEMPORAL_TASK_QUEUE",
          "value": "orchestrator-task-queue"
        },
        {
          "name": "PINECONE_INDEX",
          "value": "default-index"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:OPENAI_API_KEY::"
        },
        {
          "name": "PINECONE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:PINECONE_API_KEY::"
        },
        {
          "name": "POSTGRES_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:POSTGRES_PASSWORD::"
        },
        {
          "name": "REDIS_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:farsight/server-secrets:REDIS_PASSWORD::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/farsight-worker",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register the worker task definition:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition-worker.json
```

## Step 5: Create CloudWatch Log Groups

```bash
aws logs create-log-group --log-group-name /ecs/farsight-api-server
aws logs create-log-group --log-group-name /ecs/farsight-worker
```

## Step 6: Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name farsight-cluster
```

## Step 7: Create ECS Services

### 7.1 API Server Service

```bash
aws ecs create-service \
  --cluster farsight-cluster \
  --service-name farsight-api-server \
  --task-definition farsight-api-server \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-1>,<subnet-2>],securityGroups=[<security-group-id>],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=<target-group-arn>,containerName=api-server,containerPort=8000" \
  --health-check-grace-period-seconds 60
```

### 7.2 Worker Service

```bash
aws ecs create-service \
  --cluster farsight-cluster \
  --service-name farsight-worker \
  --task-definition farsight-worker \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-1>,<subnet-2>],securityGroups=[<security-group-id>],assignPublicIp=DISABLED}"
```

## Step 8: Set Up Application Load Balancer

### 8.1 Create Target Group

```bash
aws elbv2 create-target-group \
  --name farsight-api-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id <vpc-id> \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 10 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3
```

### 8.2 Create Load Balancer

```bash
aws elbv2 create-load-balancer \
  --name farsight-alb \
  --subnets <subnet-1> <subnet-2> \
  --security-groups <alb-security-group-id>
```

### 8.3 Create Listener

```bash
aws elbv2 create-listener \
  --load-balancer-arn <alb-arn> \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=<target-group-arn>
```

## Step 9: Configure Auto-Scaling

### 9.1 Register Scalable Targets

```bash
# API Server Auto-Scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/farsight-cluster/farsight-api-server \
  --min-capacity 2 \
  --max-capacity 10

# Worker Auto-Scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/farsight-cluster/farsight-worker \
  --min-capacity 2 \
  --max-capacity 5
```

### 9.2 Create Scaling Policies

```bash
# API Server CPU-based scaling
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/farsight-cluster/farsight-api-server \
  --policy-name api-server-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

## Step 10: Security Considerations

### 10.1 IAM Roles

Create IAM roles with the following permissions:

**ECS Task Execution Role:**
- `AmazonECSTaskExecutionRolePolicy`
- Secrets Manager read access
- ECR pull access
- CloudWatch Logs write access

**ECS Task Role:**
- Custom permissions for accessing AWS services (if needed)
- Secrets Manager read access

### 10.2 Security Groups

Configure security groups to allow:
- ALB → ECS tasks: Port 8000
- ECS tasks → RDS: Port 5432
- ECS tasks → ElastiCache: Port 6379
- ECS tasks → Temporal: Port 7233
- Internet → ALB: Port 80/443

### 10.3 Network Configuration

- Deploy ECS tasks in private subnets
- Use NAT Gateway for outbound internet access
- Place RDS and ElastiCache in private subnets
- Use VPC endpoints for AWS services to reduce NAT costs

## Step 11: Monitoring and Logging

### 11.1 CloudWatch Dashboards

Create dashboards to monitor:
- ECS service metrics (CPU, memory, task count)
- ALB metrics (request count, latency, error rates)
- RDS metrics (CPU, connections, storage)
- ElastiCache metrics (CPU, memory, evictions)

### 11.2 CloudWatch Alarms

Set up alarms for:
- High CPU utilization
- High memory utilization
- Task failures
- Health check failures
- Database connection errors

## Step 12: CI/CD Pipeline (Optional)

Set up a CI/CD pipeline using AWS CodePipeline or GitHub Actions:

1. **Build Stage**: Build Docker image on code push
2. **Test Stage**: Run unit and integration tests
3. **Push Stage**: Push image to ECR
4. **Deploy Stage**: Update ECS service with new task definition

Example GitHub Actions workflow:

```yaml
name: Deploy to ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      - name: Build and push image
        run: |
          docker build -t farsight-server ./server
          docker tag farsight-server:latest ${{ steps.login-ecr.outputs.registry }}/farsight-server:latest
          docker push ${{ steps.login-ecr.outputs.registry }}/farsight-server:latest
      - name: Update ECS service
        run: |
          aws ecs update-service --cluster farsight-cluster --service farsight-api-server --force-new-deployment
          aws ecs update-service --cluster farsight-cluster --service farsight-worker --force-new-deployment
```

## Step 13: Deployment Verification

After deployment, verify:

1. **Health Checks**: Ensure `/health` endpoint responds
2. **Logs**: Check CloudWatch logs for errors
3. **Metrics**: Monitor ECS service metrics
4. **Connectivity**: Test database and Redis connections
5. **Temporal**: Verify worker can connect to Temporal server
6. **API**: Test API endpoints through ALB

## Cost Optimization Tips

1. **Use Fargate Spot** for non-critical workloads (workers)
2. **Right-size containers** based on actual usage
3. **Enable RDS automated backups** with appropriate retention
4. **Use ElastiCache reserved instances** for predictable workloads
5. **Implement auto-scaling** to scale down during low traffic
6. **Use VPC endpoints** to reduce NAT Gateway costs

## Troubleshooting

### Common Issues

1. **Tasks failing to start**: Check CloudWatch logs, verify secrets are accessible
2. **Health check failures**: Verify security groups allow traffic, check application logs
3. **Database connection errors**: Verify RDS security group, check credentials in Secrets Manager
4. **High memory usage**: Increase task memory allocation or optimize application
5. **Worker not processing tasks**: Verify Temporal connection, check worker logs

### Useful Commands

```bash
# View service status
aws ecs describe-services --cluster farsight-cluster --services farsight-api-server

# View running tasks
aws ecs list-tasks --cluster farsight-cluster --service-name farsight-api-server

# View task logs
aws logs tail /ecs/farsight-api-server --follow

# Force new deployment
aws ecs update-service --cluster farsight-cluster --service farsight-api-server --force-new-deployment
```

## Next Steps

- Set up HTTPS with ACM certificate and ALB listener
- Configure WAF rules for security
- Set up Route 53 for custom domain
- Implement blue/green deployments
- Add container insights for detailed monitoring
- Set up backup and disaster recovery procedures

