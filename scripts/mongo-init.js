// MongoDB initialization script for Vectorless RAG
db = db.getSiblingDB('vectorless_rag');

// Create collections
db.createCollection('documents');
db.createCollection('trees');
db.createCollection('users');
db.createCollection('queries');

// Create indexes for documents collection
db.documents.createIndex({ "user_id": 1 });
db.documents.createIndex({ "status": 1 });
db.documents.createIndex({ "created_at": 1 });
db.documents.createIndex({ "title": "text", "filename": "text" });

// Create indexes for trees collection
db.trees.createIndex({ "document_id": 1 }, { unique: true });
db.trees.createIndex({ "user_id": 1 });
db.trees.createIndex({ "created_at": 1 });
db.trees.createIndex({ "updated_at": 1 });

// Create indexes for users collection
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "created_at": 1 });

// Create indexes for queries collection
db.queries.createIndex({ "user_id": 1 });
db.queries.createIndex({ "document_id": 1 });
db.queries.createIndex({ "created_at": 1 });
db.queries.createIndex({ "query_text": "text" });

print('MongoDB collections and indexes created successfully for Vectorless RAG');