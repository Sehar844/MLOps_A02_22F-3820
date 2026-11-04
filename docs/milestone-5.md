# Milestone 5 — Docker Hub Publishing and Clean-Room Deployment

## Objective

The objective of this milestone was to publish the containerized breast cancer prediction service to Docker Hub and verify that it can run on a clean environment using the published image.

## Task 16 — Publish Docker Images to Docker Hub

The Docker image was tagged and published to the public Docker Hub repository.

**Docker Hub Repository:**

https://hub.docker.com/r/sehar1515/breast-cancer-triage-api

**Published Tags:**

- `1.0.0`
- `latest`

Both tags point to the published application image.

Commands used:

```powershell
docker login
docker push sehar1515/breast-cancer-triage-api:1.0.0
docker push sehar1515/breast-cancer-triage-api:latest
```

## Task 17 — Clean-Room Deployment Verification

A separate clean-room test directory was created to verify deployment using the published Docker Hub image rather than a locally built application image.

The project image aliases were removed from the local Docker image store before the test. The clean-room Compose configuration used the published image and did not contain a local build instruction.

**Verification performed:**

1. Pulled the published API image and Redis image from their registries.
2. Started the services using Docker Compose.
3. Verified that the API and Redis containers were running.
4. Confirmed that Redis passed its health check.
5. Checked the API health endpoint.
6. Opened the Swagger documentation and submitted a prediction request.
7. Verified that the Redis prediction counter increased.
8. Inspected the pulled image's repository digest.

**Clean-Room API URL:**

`http://localhost:8003`

**Swagger Documentation:**

`http://localhost:8003/docs`

**Health Check:**

`http://localhost:8003/health`

**Statistics:**

`http://localhost:8003/stats`

## Task 18 — Written Analysis

The written analysis comparing Docker images and containers, explaining container port mapping, and describing DVC dataset restoration is included in the main assignment report.

## Outcome

The application image was successfully published to a public Docker Hub repository with both required tags. The clean-room test confirmed that the published image could be pulled and deployed without rebuilding the application from source code. The API health check, prediction endpoint, and Redis prediction counter were verified successfully.