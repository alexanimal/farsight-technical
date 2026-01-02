# Farsight Technical Infrastructure

This directory contains AWS CDK code to deploy the Farsight Technical server application to AWS ECS.

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. AWS CDK CLI installed: `npm install -g aws-cdk`
3. Python 3.10+ with dependencies installed: `uv sync` (from server directory)

## Deployment

### First-time Setup

1. Bootstrap CDK in your AWS account (if not already done):
   ```bash
   cdk bootstrap
   ```

2. Set up secrets in AWS Secrets Manager:
   The stack creates a secret named `farsight/server-secrets` but you need to populate it with actual values:
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id farsight/server-secrets \
     --secret-string '{
       "OPENAI_API_KEY": "your-openai-key",
       "PINECONE_API_KEY": "your-pinecone-key",
       "POSTGRES_PASSWORD": "your-postgres-password",
       "REDIS_PASSWORD": "your-redis-password",
       "API_KEY": "your-api-key"
     }'
   ```

### Deploy Infrastructure

From the `server` directory:

```bash
# Install dependencies
uv sync

# Synthesize CloudFormation template (optional, for review)
cdk synth

# Deploy the stack
cdk deploy

# Or deploy with specific context values
cdk deploy --context region=us-east-1 --context vpc_cidr=10.0.0.0/16
```

### Configuration Context

You can configure the stack using CDK context:

```bash
# Set region
cdk deploy --context region=us-east-1

# Set VPC CIDR
cdk deploy --context vpc_cidr=10.0.0.0/16

# Set database instance class
cdk deploy --context db_instance_class=db.t3.medium

# Set Redis node type
cdk deploy --context redis_node_type=cache.t3.micro

# Set ECR repository name
cdk deploy --context ecr_repo_name=farsight-server

# Set cluster name
cdk deploy --context cluster_name=farsight-cluster

# Set Temporal address (if using external Temporal)
cdk deploy --context temporal_address=temporal.example.com:7233
```

### Destroy Infrastructure

```bash
cdk destroy
```

**Warning:** This will delete all resources including databases. Make sure you have backups!

## What Gets Created

The CDK stack creates:

1. **VPC** with public and private subnets across 2 AZs
2. **ECR Repository** for Docker images
3. **RDS PostgreSQL** database instance (db.t3.medium, Multi-AZ)
4. **ElastiCache Redis** cluster (cache.t3.micro)
5. **Secrets Manager** secret for application secrets
6. **IAM Roles** for ECS task execution and task roles
7. **CloudWatch Log Groups** for API and worker services
8. **ECS Cluster** (Fargate)
9. **ECS Task Definitions** for API server and worker
10. **ECS Services** for API server and worker
11. **Application Load Balancer** with target group and listener
12. **Auto-scaling** policies for both services
13. **CloudWatch Alarms** for monitoring
14. **Security Groups** with appropriate rules
15. **VPC Endpoints** for ECR, CloudWatch Logs, and Secrets Manager

## Outputs

After deployment, the stack outputs:

- `LoadBalancerDNS`: The DNS name of the Application Load Balancer
- `ECRRepositoryURI`: The URI of the ECR repository for pushing images

## CI/CD Integration

The infrastructure is designed to work with the GitHub Actions workflow in `.github/workflows/cd.yml`. The workflow:

1. Builds and pushes Docker images to ECR
2. Deploys infrastructure with CDK
3. Updates ECS services to use the new images

## Manual Image Updates

To manually update the ECS services with a new image:

```bash
# Build and push new image
docker build -t farsight-server:latest ./server
docker tag farsight-server:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/farsight-server:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/farsight-server:latest

# Force new deployment
aws ecs update-service \
  --cluster farsight-cluster \
  --service farsight-api-server \
  --force-new-deployment

aws ecs update-service \
  --cluster farsight-cluster \
  --service farsight-worker \
  --force-new-deployment
```

## Troubleshooting

### CDK Deployment Fails

- Check AWS credentials: `aws sts get-caller-identity`
- Ensure CDK is bootstrapped: `cdk bootstrap`
- Check CloudFormation console for detailed error messages

### ECS Tasks Not Starting

- Check CloudWatch logs: `/ecs/farsight-api-server` and `/ecs/farsight-worker`
- Verify secrets are accessible: Check IAM role permissions
- Check security group rules allow necessary traffic
- Verify VPC endpoints are working (if using private subnets)

### Database Connection Issues

- Verify RDS security group allows traffic from ECS security group on port 5432
- Check RDS endpoint is correct in task definition
- Verify database credentials in Secrets Manager

### Redis Connection Issues

- Verify ElastiCache security group allows traffic from ECS security group on port 6379
- Check Redis endpoint is correct in task definition
- Verify Redis is in the same VPC as ECS tasks

## Cost Considerations

- **RDS Multi-AZ**: Doubles database costs but provides high availability
- **NAT Gateway**: ~$32/month + data transfer costs
- **VPC Endpoints**: Can reduce NAT Gateway costs but have their own costs
- **ECS Fargate**: Pay per vCPU and memory used
- **ALB**: ~$16/month + LCU costs
- **ElastiCache**: Pay per node type

Consider:
- Using single-AZ RDS for development
- Using Fargate Spot for workers (non-critical workloads)
- Right-sizing instances based on actual usage
- Using VPC endpoints to reduce NAT Gateway data transfer costs

