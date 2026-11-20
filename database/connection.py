from pymongo import MongoClient
from datetime import datetime
import os
from bson.objectid import ObjectId

class DatabaseManager:
    """MongoDB connection and operations manager"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.client = MongoClient(mongo_uri)
        self.db = self.client['underwater_ai_db']
        self.collection = self.db['analysis_results']
        
        # Create index on timestamp for efficient querying
        self.collection.create_index('timestamp')
        self.collection.create_index('unique_id')
    
    def insert_record(self, record):
        """
        Insert analysis record into database
        
        Args:
            record: Dictionary containing analysis results
        
        Returns:
            Inserted document ID
        """
        result = self.collection.insert_one(record)
        return result.inserted_id
    
    def get_record(self, unique_id):
        """
        Retrieve specific analysis record
        
        Args:
            unique_id: Unique identifier for the analysis
        
        Returns:
            Record dictionary or None
        """
        record = self.collection.find_one({'unique_id': unique_id})
        if record:
            record['_id'] = str(record['_id'])
            record['timestamp'] = record['timestamp'].isoformat()
        return record
    
    def get_all_records(self, limit=50):
        """
        Retrieve all analysis records (most recent first)
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of record dictionaries
        """
        records = list(self.collection.find().sort('timestamp', -1).limit(limit))
        for record in records:
            record['_id'] = str(record['_id'])
            record['timestamp'] = record['timestamp'].isoformat()
        return records
    
    def delete_record(self, unique_id):
        """
        Delete analysis record
        
        Args:
            unique_id: Unique identifier for the analysis
        
        Returns:
            Deleted count
        """
        result = self.collection.delete_one({'unique_id': unique_id})
        return result.deleted_count
    
    def update_record(self, unique_id, update_data):
        """
        Update analysis record
        
        Args:
            unique_id: Unique identifier for the analysis
            update_data: Dictionary with fields to update
        
        Returns:
            Modified count
        """
        result = self.collection.update_one(
            {'unique_id': unique_id},
            {'$set': update_data}
        )
        return result.modified_count
