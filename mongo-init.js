// Admin user
admin = db.getSiblingDB('admin');
admin.createUser({
  user: 'FC-DB-ADMIN',
  pwd: 'CHANGE-ME-ADMIN',
  roles: ['root']
});

// DB Init
db = db.getSiblingDB('fact_checker');
db.createCollection('embeddings');

// Index Init
db.embeddings.createIndex({ "metadata.source": 1 });
db.embeddings.createIndex({ "embedding": "text" });

// Logs
print("Database 'fact_checker' initialized successfully");
print("Collections created: embeddings");
