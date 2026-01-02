"""Main CDK Stack for Farsight Technical infrastructure.

This stack creates all AWS resources needed to deploy the Farsight server
application to ECS, including:
- VPC with public/private subnets
- ECR repository
- RDS PostgreSQL database
- ElastiCache Redis cluster
- Secrets Manager secret
- IAM roles
- CloudWatch log groups
- ECS cluster, task definitions, and services
- Application Load Balancer
- Auto-scaling configuration
- Security groups and networking
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_rds as rds,
    aws_elasticache as elasticache,
    aws_secretsmanager as secretsmanager,
    aws_iam as iam,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
)
from constructs import Construct


class FarsightStack(cdk.Stack):
    """Main stack for Farsight Technical infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Configuration from context or defaults
        self.vpc_cidr = self.node.try_get_context("vpc_cidr") or "10.0.0.0/16"
        self.db_instance_class = self.node.try_get_context("db_instance_class") or "db.t3.medium"
        self.redis_node_type = self.node.try_get_context("redis_node_type") or "cache.t3.micro"
        self.ecr_repo_name = self.node.try_get_context("ecr_repo_name") or "farsight-server"
        self.cluster_name = self.node.try_get_context("cluster_name") or "farsight-cluster"
        self.temporal_address = self.node.try_get_context("temporal_address") or "temporal:7233"

        # Create VPC and networking
        self.vpc = self._create_vpc()
        self.ecr_repository = self._create_ecr_repository()
        self.secret = self._create_secrets_manager_secret()
        self.db = self._create_rds_database()
        self.redis = self._create_elasticache_cluster()
        self.iam_roles = self._create_iam_roles()
        self.log_groups = self._create_log_groups()
        self.cluster = self._create_ecs_cluster()
        self.alb = self._create_application_load_balancer()
        self.api_service = self._create_api_service()
        self.worker_service = self._create_worker_service()
        self._setup_auto_scaling()
        self._setup_monitoring()

    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public and private subnets."""
        vpc = ec2.Vpc(
            self,
            "FarsightVPC",
            ip_addresses=ec2.IpAddresses.cidr(self.vpc_cidr),
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    name="Public",
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name="Private",
                    cidr_mask=24,
                ),
            ],
        )

        # Add VPC endpoint for ECR to reduce NAT Gateway costs
        vpc.add_gateway_endpoint(
            "ECREndpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        vpc.add_interface_endpoint(
            "ECRDockerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
        )

        vpc.add_interface_endpoint(
            "ECRAPIEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR,
        )

        vpc.add_interface_endpoint(
            "CloudWatchLogsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
        )

        vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
        )

        return vpc

    def _create_ecr_repository(self) -> ecr.Repository:
        """Create ECR repository for Docker images."""
        return ecr.Repository(
            self,
            "ECRRepository",
            repository_name=self.ecr_repo_name,
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep last 10 images",
                    max_image_count=10,
                )
            ],
        )

    def _create_secrets_manager_secret(self) -> secretsmanager.Secret:
        """Create Secrets Manager secret for application secrets."""
        return secretsmanager.Secret(
            self,
            "AppSecrets",
            secret_name="farsight/server-secrets",
            description="Secrets for Farsight server application",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"OPENAI_API_KEY":"","PINECONE_API_KEY":"","POSTGRES_PASSWORD":"","REDIS_PASSWORD":"","API_KEY":""}',
                generate_string_key="DUMMY",
                exclude_characters='",\\',
            ),
        )

    def _create_rds_database(self) -> rds.DatabaseInstance:
        """Create RDS PostgreSQL database instance."""
        # Security group for RDS
        db_security_group = ec2.SecurityGroup(
            self,
            "RDSSecurityGroup",
            vpc=self.vpc,
            description="Security group for RDS PostgreSQL",
            allow_all_outbound=False,
        )

        # Subnet group for RDS
        subnet_group = rds.SubnetGroup(
            self,
            "DBSubnetGroup",
            vpc=self.vpc,
            description="Subnet group for RDS",
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        )

        # Create database instance
        database = rds.DatabaseInstance(
            self,
            "PostgreSQLDatabase",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_14_9),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            subnet_group=subnet_group,
            security_groups=[db_security_group],
            database_name="farsight",
            credentials=rds.Credentials.from_generated_secret(
                "postgres", exclude_characters='",\\'
            ),
            allocated_storage=100,
            backup_retention=cdk.Duration.days(7),
            multi_az=True,
            deletion_protection=False,  # Set to True in production
            removal_policy=cdk.RemovalPolicy.DESTROY,  # Change to RETAIN in production
        )

        # Store reference to security group for later use
        self.db_security_group = db_security_group

        return database

    def _create_elasticache_cluster(self) -> elasticache.CfnCacheCluster:
        """Create ElastiCache Redis cluster."""
        # Security group for ElastiCache
        redis_security_group = ec2.SecurityGroup(
            self,
            "RedisSecurityGroup",
            vpc=self.vpc,
            description="Security group for ElastiCache Redis",
            allow_all_outbound=False,
        )

        # Subnet group for ElastiCache
        subnet_group = elasticache.CfnSubnetGroup(
            self,
            "RedisSubnetGroup",
            description="Subnet group for ElastiCache",
            subnet_ids=[subnet.subnet_id for subnet in self.vpc.private_subnets],
        )

        # Create Redis cluster
        redis_cluster = elasticache.CfnCacheCluster(
            self,
            "RedisCluster",
            cache_node_type=self.redis_node_type,
            engine="redis",
            num_cache_nodes=1,
            vpc_security_group_ids=[redis_security_group.security_group_id],
            cache_subnet_group_name=subnet_group.ref,
        )

        # Store reference to security group for later use
        self.redis_security_group = redis_security_group

        return redis_cluster

    def _create_iam_roles(self) -> dict[str, iam.Role]:
        """Create IAM roles for ECS tasks."""
        # Task execution role
        execution_role = iam.Role(
            self,
            "ECSTaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )

        # Grant permissions to pull from ECR
        self.ecr_repository.grant_pull(execution_role)

        # Grant permissions to read secrets
        self.secret.grant_read(execution_role)

        # Grant permissions to write logs
        execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["*"],
            )
        )

        # Task role (for application to access AWS services)
        task_role = iam.Role(
            self,
            "ECSTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # Grant permissions to read secrets
        self.secret.grant_read(task_role)

        return {
            "execution_role": execution_role,
            "task_role": task_role,
        }

    def _create_log_groups(self) -> dict[str, logs.LogGroup]:
        """Create CloudWatch log groups."""
        api_log_group = logs.LogGroup(
            self,
            "APILogGroup",
            log_group_name="/ecs/farsight-api-server",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        worker_log_group = logs.LogGroup(
            self,
            "WorkerLogGroup",
            log_group_name="/ecs/farsight-worker",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        return {
            "api": api_log_group,
            "worker": worker_log_group,
        }

    def _create_ecs_cluster(self) -> ecs.Cluster:
        """Create ECS cluster."""
        return ecs.Cluster(
            self,
            "ECSCluster",
            cluster_name=self.cluster_name,
            vpc=self.vpc,
            container_insights=True,
        )

    def _create_application_load_balancer(self) -> elbv2.ApplicationLoadBalancer:
        """Create Application Load Balancer."""
        # Security group for ALB
        alb_security_group = ec2.SecurityGroup(
            self,
            "ALBSecurityGroup",
            vpc=self.vpc,
            description="Security group for Application Load Balancer",
            allow_all_outbound=True,
        )

        # Allow HTTP traffic from internet
        alb_security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "Allow HTTP from internet",
        )

        # Create ALB
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "ApplicationLoadBalancer",
            vpc=self.vpc,
            internet_facing=True,
            security_group=alb_security_group,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        # Store reference to security group for later use
        self.alb_security_group = alb_security_group

        return alb

    def _create_api_service(self) -> ecs.FargateService:
        """Create ECS Fargate service for API server."""
        # Security group for ECS tasks
        ecs_security_group = ec2.SecurityGroup(
            self,
            "ECSSecurityGroup",
            vpc=self.vpc,
            description="Security group for ECS tasks",
            allow_all_outbound=True,
        )

        # Allow traffic from ALB
        ecs_security_group.add_ingress_rule(
            self.alb_security_group,
            ec2.Port.tcp(8000),
            "Allow traffic from ALB",
        )

        # Allow traffic to RDS
        self.db_security_group.add_ingress_rule(
            ecs_security_group,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL from ECS tasks",
        )

        # Allow traffic to Redis
        self.redis_security_group.add_ingress_rule(
            ecs_security_group,
            ec2.Port.tcp(6379),
            "Allow Redis from ECS tasks",
        )

        # Task definition
        task_definition = ecs.FargateTaskDefinition(
            self,
            "APITaskDefinition",
            cpu=512,
            memory_limit_mib=1024,
            execution_role=self.iam_roles["execution_role"],
            task_role=self.iam_roles["task_role"],
        )

        # Container definition
        container = task_definition.add_container(
            "api-server",
            image=ecs.ContainerImage.from_registry(
                f"{cdk.Aws.ACCOUNT_ID}.dkr.ecr.{cdk.Aws.REGION}.amazonaws.com/{self.ecr_repo_name}:latest"
            ),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="ecs",
                log_group=self.log_groups["api"],
            ),
            environment={
                "POSTGRES_HOST": self.db.instance_endpoint.hostname,
                "POSTGRES_PORT": "5432",
                "POSTGRES_DB_NAME": "farsight",
                "POSTGRES_USER": "postgres",
                "REDIS_HOST": self.redis.attr_redis_endpoint_address,
                "REDIS_PORT": "6379",
                "TEMPORAL_ADDRESS": self.temporal_address,
                "TEMPORAL_NAMESPACE": "default",
                "TEMPORAL_TASK_QUEUE": "orchestrator-task-queue",
                "PINECONE_INDEX": "default-index",
            },
            secrets={
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(self.secret, "OPENAI_API_KEY"),
                "PINECONE_API_KEY": ecs.Secret.from_secrets_manager(
                    self.secret, "PINECONE_API_KEY"
                ),
                "POSTGRES_PASSWORD": ecs.Secret.from_secrets_manager(self.db.secret, "password"),
                "REDIS_PASSWORD": ecs.Secret.from_secrets_manager(self.secret, "REDIS_PASSWORD"),
                "API_KEY": ecs.Secret.from_secrets_manager(self.secret, "API_KEY"),
            },
        )

        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8000,
                protocol=ecs.Protocol.TCP,
            )
        )

        # Health check
        container.add_health_check(
            ecs.HealthCheck(
                command=[
                    "CMD-SHELL",
                    "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1",
                ],
                interval=cdk.Duration.seconds(30),
                timeout=cdk.Duration.seconds(10),
                retries=3,
                start_period=cdk.Duration.seconds(60),
            )
        )

        # Target group
        target_group = elbv2.ApplicationTargetGroup(
            self,
            "APITargetGroup",
            vpc=self.vpc,
            port=8000,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=cdk.Duration.seconds(30),
                timeout=cdk.Duration.seconds(10),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # Create listener
        self.alb.add_listener(
            "HTTPListener",
            port=80,
            protocol=elbv2.ApplicationProtocol.HTTP,
            default_target_groups=[target_group],
        )

        # Create service
        service = ecs.FargateService(
            self,
            "APIService",
            cluster=self.cluster,
            task_definition=task_definition,
            desired_count=2,
            security_groups=[ecs_security_group],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            assign_public_ip=False,
            health_check_grace_period=cdk.Duration.seconds(60),
        )

        # Attach service to target group
        service.attach_to_application_target_group(target_group)

        return service

    def _create_worker_service(self) -> ecs.FargateService:
        """Create ECS Fargate service for Temporal worker."""
        # Reuse the same security group as API service
        ecs_security_group = self.ecs_security_group

        # Allow traffic to RDS (if not already allowed)
        self.db_security_group.add_ingress_rule(
            ecs_security_group,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL from worker tasks",
        )

        # Allow traffic to Redis (if not already allowed)
        self.redis_security_group.add_ingress_rule(
            ecs_security_group,
            ec2.Port.tcp(6379),
            "Allow Redis from worker tasks",
        )

        # Task definition
        task_definition = ecs.FargateTaskDefinition(
            self,
            "WorkerTaskDefinition",
            cpu=512,
            memory_limit_mib=1024,
            execution_role=self.iam_roles["execution_role"],
            task_role=self.iam_roles["task_role"],
        )

        # Container definition
        container = task_definition.add_container(
            "temporal-worker",
            image=ecs.ContainerImage.from_registry(
                f"{cdk.Aws.ACCOUNT_ID}.dkr.ecr.{cdk.Aws.REGION}.amazonaws.com/{self.ecr_repo_name}:latest"
            ),
            command=["python", "-m", "src.temporal.worker"],
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="ecs",
                log_group=self.log_groups["worker"],
            ),
            environment={
                "POSTGRES_HOST": self.db.instance_endpoint.hostname,
                "POSTGRES_PORT": "5432",
                "POSTGRES_DB_NAME": "farsight",
                "POSTGRES_USER": "postgres",
                "REDIS_HOST": self.redis.attr_redis_endpoint_address,
                "REDIS_PORT": "6379",
                "TEMPORAL_ADDRESS": self.temporal_address,
                "TEMPORAL_NAMESPACE": "default",
                "TEMPORAL_TASK_QUEUE": "orchestrator-task-queue",
                "PINECONE_INDEX": "default-index",
            },
            secrets={
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(self.secret, "OPENAI_API_KEY"),
                "PINECONE_API_KEY": ecs.Secret.from_secrets_manager(
                    self.secret, "PINECONE_API_KEY"
                ),
                "POSTGRES_PASSWORD": ecs.Secret.from_secrets_manager(self.db.secret, "password"),
                "REDIS_PASSWORD": ecs.Secret.from_secrets_manager(self.secret, "REDIS_PASSWORD"),
            },
        )

        # Create service
        service = ecs.FargateService(
            self,
            "WorkerService",
            cluster=self.cluster,
            task_definition=task_definition,
            desired_count=2,
            security_groups=[ecs_security_group],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            assign_public_ip=False,
        )

        return service

    def _setup_auto_scaling(self) -> None:
        """Configure auto-scaling for ECS services."""
        # API service auto-scaling
        api_scaling = self.api_service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10,
        )

        api_scaling.scale_on_cpu_utilization(
            "APICPUScaling",
            target_utilization_percent=70,
            scale_in_cooldown=cdk.Duration.seconds(300),
            scale_out_cooldown=cdk.Duration.seconds(60),
        )

        # Worker service auto-scaling
        worker_scaling = self.worker_service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=5,
        )

        worker_scaling.scale_on_cpu_utilization(
            "WorkerCPUScaling",
            target_utilization_percent=70,
            scale_in_cooldown=cdk.Duration.seconds(300),
            scale_out_cooldown=cdk.Duration.seconds(60),
        )

    def _setup_monitoring(self) -> None:
        """Set up CloudWatch alarms and monitoring."""
        # SNS topic for alarms
        alarm_topic = sns.Topic(
            self,
            "AlarmTopic",
            display_name="Farsight Alarms",
        )

        # API service alarms
        api_cpu_alarm = cloudwatch.Alarm(
            self,
            "APICPUAlarm",
            metric=self.api_service.metric_cpu_utilization(),
            threshold=80,
            evaluation_periods=2,
            alarm_description="API service CPU utilization is high",
        )

        api_memory_alarm = cloudwatch.Alarm(
            self,
            "APIMemoryAlarm",
            metric=self.api_service.metric_memory_utilization(),
            threshold=80,
            evaluation_periods=2,
            alarm_description="API service memory utilization is high",
        )

        # Worker service alarms
        worker_cpu_alarm = cloudwatch.Alarm(
            self,
            "WorkerCPUAlarm",
            metric=self.worker_service.metric_cpu_utilization(),
            threshold=80,
            evaluation_periods=2,
            alarm_description="Worker service CPU utilization is high",
        )

        # Add alarm actions
        api_cpu_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))
        api_memory_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))
        worker_cpu_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))

        # Output ALB DNS name
        cdk.CfnOutput(
            self,
            "LoadBalancerDNS",
            value=self.alb.load_balancer_dns_name,
            description="Application Load Balancer DNS name",
        )

        # Output ECR repository URI
        cdk.CfnOutput(
            self,
            "ECRRepositoryURI",
            value=self.ecr_repository.repository_uri,
            description="ECR repository URI for pushing Docker images",
        )
