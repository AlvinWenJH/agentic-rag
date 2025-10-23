#!/usr/bin/env python3
"""
Live API Test Script for Vectorless RAG Backend
Tests the actual running FastAPI container to verify what's working and what's failing.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class APITester:
    """Live API testing class for the Vectorless RAG backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        print(f"{status} {test_name}: {message}")
        if details and not success:
            print(f"   Details: {details}")
    
    def test_server_connectivity(self) -> bool:
        """Test basic server connectivity."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            if response.status_code in [200, 404, 422]:  # Any response means server is up
                self.log_result(
                    "Server Connectivity", 
                    True, 
                    f"Server is responding (status: {response.status_code})"
                )
                return True
            else:
                self.log_result(
                    "Server Connectivity", 
                    False, 
                    f"Unexpected status code: {response.status_code}",
                    {"status_code": response.status_code, "response": response.text[:200]}
                )
                return False
        except requests.exceptions.ConnectionError:
            self.log_result(
                "Server Connectivity", 
                False, 
                "Cannot connect to server - is it running?",
                {"url": self.base_url}
            )
            return False
        except requests.exceptions.Timeout:
            self.log_result(
                "Server Connectivity", 
                False, 
                f"Server timeout after {self.timeout}s"
            )
            return False
        except Exception as e:
            self.log_result(
                "Server Connectivity", 
                False, 
                f"Connection error: {str(e)}"
            )
            return False
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                expected_fields = ["status", "service", "version"]
                missing_fields = [field for field in expected_fields if field not in data]
                
                if not missing_fields and data.get("status") == "healthy":
                    self.log_result(
                        "Health Endpoint", 
                        True, 
                        "Health check passed",
                        {"response": data}
                    )
                else:
                    self.log_result(
                        "Health Endpoint", 
                        False, 
                        f"Health check incomplete. Missing: {missing_fields}",
                        {"response": data}
                    )
            else:
                self.log_result(
                    "Health Endpoint", 
                    False, 
                    f"Health endpoint returned {response.status_code}",
                    {"status_code": response.status_code, "response": response.text[:200]}
                )
        except Exception as e:
            self.log_result(
                "Health Endpoint", 
                False, 
                f"Health endpoint error: {str(e)}"
            )
    
    def test_docs_endpoint(self):
        """Test the API documentation endpoint."""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=self.timeout)
            if response.status_code == 200:
                if "swagger" in response.text.lower() or "openapi" in response.text.lower():
                    self.log_result(
                        "API Docs", 
                        True, 
                        "API documentation is accessible"
                    )
                else:
                    self.log_result(
                        "API Docs", 
                        False, 
                        "Docs endpoint accessible but content unexpected"
                    )
            else:
                self.log_result(
                    "API Docs", 
                    False, 
                    f"Docs endpoint returned {response.status_code}",
                    {"status_code": response.status_code}
                )
        except Exception as e:
            self.log_result(
                "API Docs", 
                False, 
                f"Docs endpoint error: {str(e)}"
            )
    
    def test_openapi_schema(self):
        """Test the OpenAPI schema endpoint."""
        try:
            response = requests.get(f"{self.base_url}/openapi.json", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if "openapi" in data and "paths" in data:
                    paths_count = len(data.get("paths", {}))
                    self.log_result(
                        "OpenAPI Schema", 
                        True, 
                        f"OpenAPI schema available with {paths_count} endpoints"
                    )
                else:
                    self.log_result(
                        "OpenAPI Schema", 
                        False, 
                        "OpenAPI schema format invalid"
                    )
            else:
                self.log_result(
                    "OpenAPI Schema", 
                    False, 
                    f"OpenAPI schema returned {response.status_code}"
                )
        except Exception as e:
            self.log_result(
                "OpenAPI Schema", 
                False, 
                f"OpenAPI schema error: {str(e)}"
            )
    
    def test_documents_endpoint(self):
        """Test the documents API endpoint."""
        try:
            # Test GET /api/v1/documents/
            response = requests.get(f"{self.base_url}/api/v1/documents/", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "documents" in data:
                    self.log_result(
                        "Documents List", 
                        True, 
                        f"Documents endpoint working, found {len(data.get('documents', []))} documents"
                    )
                else:
                    self.log_result(
                        "Documents List", 
                        False, 
                        "Documents endpoint returned unexpected format",
                        {"response": data}
                    )
            elif response.status_code == 500:
                self.log_result(
                    "Documents List", 
                    False, 
                    "Documents endpoint has internal server error (500) - likely database issue"
                )
            else:
                self.log_result(
                    "Documents List", 
                    False, 
                    f"Documents endpoint returned {response.status_code}",
                    {"status_code": response.status_code, "response": response.text[:200]}
                )
        except Exception as e:
            self.log_result(
                "Documents List", 
                False, 
                f"Documents endpoint error: {str(e)}"
            )
    
    def test_trees_endpoint(self):
        """Test the trees API endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/trees/", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "trees" in data:
                    self.log_result(
                        "Trees List", 
                        True, 
                        f"Trees endpoint working, found {len(data.get('trees', []))} trees"
                    )
                else:
                    self.log_result(
                        "Trees List", 
                        False, 
                        "Trees endpoint returned unexpected format"
                    )
            elif response.status_code == 500:
                self.log_result(
                    "Trees List", 
                    False, 
                    "Trees endpoint has internal server error (500) - likely database issue"
                )
            else:
                self.log_result(
                    "Trees List", 
                    False, 
                    f"Trees endpoint returned {response.status_code}"
                )
        except Exception as e:
            self.log_result(
                "Trees List", 
                False, 
                f"Trees endpoint error: {str(e)}"
            )
    
    def test_queries_endpoint(self):
        """Test the queries API endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/queries/", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "queries" in data:
                    self.log_result(
                        "Queries List", 
                        True, 
                        f"Queries endpoint working, found {len(data.get('queries', []))} queries"
                    )
                else:
                    self.log_result(
                        "Queries List", 
                        False, 
                        "Queries endpoint returned unexpected format"
                    )
            elif response.status_code == 500:
                self.log_result(
                    "Queries List", 
                    False, 
                    "Queries endpoint has internal server error (500) - likely database issue"
                )
            else:
                self.log_result(
                    "Queries List", 
                    False, 
                    f"Queries endpoint returned {response.status_code}"
                )
        except Exception as e:
            self.log_result(
                "Queries List", 
                False, 
                f"Queries endpoint error: {str(e)}"
            )
    
    def test_users_endpoint(self):
        """Test the users API endpoint."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/users/", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "users" in data:
                    self.log_result(
                        "Users List", 
                        True, 
                        f"Users endpoint working, found {len(data.get('users', []))} users"
                    )
                else:
                    self.log_result(
                        "Users List", 
                        False, 
                        "Users endpoint returned unexpected format"
                    )
            elif response.status_code == 500:
                self.log_result(
                    "Users List", 
                    False, 
                    "Users endpoint has internal server error (500) - likely database issue"
                )
            else:
                self.log_result(
                    "Users List", 
                    False, 
                    f"Users endpoint returned {response.status_code}"
                )
        except Exception as e:
            self.log_result(
                "Users List", 
                False, 
                f"Users endpoint error: {str(e)}"
            )
    
    def test_database_connectivity(self):
        """Test database connectivity by checking if endpoints return 500 errors."""
        endpoints_with_500 = []
        for result in self.results:
            if "500" in result["message"] or "internal server error" in result["message"].lower():
                endpoints_with_500.append(result["test"])
        
        if endpoints_with_500:
            self.log_result(
                "Database Connectivity", 
                False, 
                f"Multiple endpoints returning 500 errors: {', '.join(endpoints_with_500)}. This suggests database connection issues."
            )
        else:
            self.log_result(
                "Database Connectivity", 
                True, 
                "No widespread 500 errors detected - database likely connected"
            )
    
    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting Live API Tests for Vectorless RAG Backend")
        print(f"📍 Testing server at: {self.base_url}")
        print(f"⏱️  Timeout: {self.timeout}s")
        print("=" * 60)
        
        # Test basic connectivity first
        if not self.test_server_connectivity():
            print("\n❌ Server is not accessible. Please check if the backend container is running.")
            print("   Try: docker-compose up -d")
            return
        
        # Run all tests
        self.test_health_endpoint()
        self.test_docs_endpoint()
        self.test_openapi_schema()
        self.test_documents_endpoint()
        self.test_trees_endpoint()
        self.test_queries_endpoint()
        self.test_users_endpoint()
        self.test_database_connectivity()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        
        if passed == total:
            print("\n🎉 All tests passed! Your API is working correctly.")
        elif passed > total // 2:
            print("\n⚠️  Most tests passed, but some issues detected.")
        else:
            print("\n🚨 Multiple issues detected. Check the failures above.")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        failed_tests = [r for r in self.results if not r["success"]]
        
        if any("500" in r["message"] for r in failed_tests):
            print("   • Check database connection (MongoDB)")
            print("   • Verify environment variables in .env file")
            print("   • Check container logs: docker-compose logs vectorless-rag-api")
        
        if any("connect" in r["message"].lower() for r in failed_tests):
            print("   • Start the backend: docker-compose up -d")
            print("   • Check if port 8000 is available")
        
        if any("timeout" in r["message"].lower() for r in failed_tests):
            print("   • Server may be starting up - wait a moment and retry")
            print("   • Check server performance and resources")
        
        print(f"\n📝 Detailed results saved to test results above")
        return self.results


def main():
    """Main function to run the API tests."""
    # You can customize the base URL and timeout here
    base_url = "http://localhost:8000"
    timeout = 10
    
    # Check if custom URL provided as argument
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = APITester(base_url=base_url, timeout=timeout)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(failed_count)


if __name__ == "__main__":
    main()