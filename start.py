#!/usr/bin/env python3
"""
Development startup script for Vectorless RAG system.
This script helps with local development setup and testing.
"""

import asyncio
import os
import sys
import subprocess


def check_requirements():
    """Check if all requirements are installed."""
    try:
        import fastapi
        import motor
        import redis
        import minio
        import google.generativeai

        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def check_environment():
    """Check if environment variables are set."""
    required_vars = ["GEMINI_API_KEY", "MONGODB_URL", "REDIS_URL", "MINIO_ENDPOINT"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file based on .env.example")
        return False

    print("✅ Environment variables are set")
    return True


async def check_services():
    """Check if external services are running."""
    import motor.motor_asyncio
    import redis.asyncio as redis
    import minio

    # Check MongoDB
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGODB_URL"))
        await client.admin.command("ping")
        print("✅ MongoDB is running")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

    # Check Redis
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL"))
        await redis_client.ping()
        print("✅ Redis is running")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

    # Check MinIO
    try:
        minio_client = minio.Minio(
            os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
        # Try to list buckets
        list(minio_client.list_buckets())
        print("✅ MinIO is running")
    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")
        return False

    return True


def start_docker_services():
    """Start Docker services if not running."""
    print("🐳 Starting Docker services...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✅ Docker services started")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Docker services: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker Compose not found. Please install Docker and Docker Compose")
        return False


def start_application():
    """Start the FastAPI application."""
    print("🚀 Starting Vectorless RAG application...")
    try:
        subprocess.run(
            [
                "uvicorn",
                "app.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")


async def main():
    """Main startup function."""
    print("🔧 Vectorless RAG Development Setup")
    print("=" * 40)

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Start Docker services if needed
    services_running = await check_services()
    if not services_running:
        print("🔄 Services not running, starting Docker services...")
        if not start_docker_services():
            sys.exit(1)

        # Wait a bit for services to start
        print("⏳ Waiting for services to start...")
        await asyncio.sleep(10)

        # Check again
        if not await check_services():
            print("❌ Services still not running. Please check Docker logs.")
            sys.exit(1)

    print("\n✅ All checks passed! Starting application...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("📊 MinIO Console: http://localhost:9001")
    print("\nPress Ctrl+C to stop the application")
    print("=" * 40)

    # Start the application
    start_application()


if __name__ == "__main__":
    asyncio.run(main())
