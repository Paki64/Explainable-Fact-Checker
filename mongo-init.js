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
db.embeddings.createIndex({ "embedding": 1 }, { sparse: true });
db.embeddings.createIndex({ "metadata.url": 1 });
db.embeddings.createIndex({ "metadata.source": 1 });

// Logs
print("Database 'fact_checker' initialized successfully");
print("Collections created: embeddings");
print("Indexes created: embedding, metadata.url, metadata.source");
db.embeddings.createIndex({ "embedding": 1 }, { sparse: true });
