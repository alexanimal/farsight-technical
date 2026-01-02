#!/usr/bin/env python3
"""CDK App entry point for Farsight Technical infrastructure."""

import aws_cdk as cdk
from infrastructure.stack import FarsightStack

app = cdk.App()

FarsightStack(
    app,
    "FarsightStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or None,
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    description="Farsight Technical ECS deployment infrastructure",
)

app.synth()
