import hashlib
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.database import get_connection_string

logger = logging.getLogger(__name__)

class DatabaseShardManager:
    """Manage database sharding for large user base"""
    
    def __init__(self, num_shards=4):
        self.num_shards = num_shards
        self.engines = {}
        self.sessions = {}
        
        # Initialize connections to all shards
        for shard_id in range(num_shards):
            connection_string = get_connection_string(shard_id)
            self.engines[shard_id] = create_engine(connection_string)
            self.sessions[shard_id] = sessionmaker(bind=self.engines[shard_id])
    
    def get_shard_id(self, user_id):
        """Determine shard ID for a user"""
        # Hash user_id to determine shard
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return hash_value % self.num_shards
    
    def get_session(self, user_id):
        """Get database session for a user"""
        shard_id = self.get_shard_id(user_id)
        return self.sessions[shard_id]()
    
    def execute_on_all_shards(self, query_func):
        """Execute a query function on all shards"""
        results = []
        for shard_id in range(self.num_shards):
            session = self.sessions[shard_id]()
            try:
                result = query_func(session)
                results.append(result)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Error executing query on shard {shard_id}: {e}")
            finally:
                session.close()
        return results
    
    def get_user_data(self, user_id, model_class):
        """Get user-specific data from the appropriate shard"""
        session = self.get_session(user_id)
        try:
            result = session.query(model_class).filter_by(user_id=user_id).all()
            return result
        except Exception as e:
            logger.error(f"Error fetching user data for {user_id}: {e}")
            return []
        finally:
            session.close()
    
    def save_user_data(self, user_id, model_instance):
        """Save user-specific data to the appropriate shard"""
        session = self.get_session(user_id)
        try:
            session.add(model_instance)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving user data for {user_id}: {e}")
            return False
        finally:
            session.close()
    
    def batch_save(self, user_id, model_instances):
        """Save multiple model instances for a user in a batch"""
        session = self.get_session(user_id)
        try:
            session.add_all(model_instances)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error batch saving for user {user_id}: {e}")
            return False
        finally:
            session.close()
    
    def initialize_shards(self, schema_creation_func):
        """Initialize schema on all shards"""
        for shard_id in range(self.num_shards):
            engine = self.engines[shard_id]
            try:
                schema_creation_func(engine)
                logger.info(f"Initialized schema on shard {shard_id}")
            except Exception as e:
                logger.error(f"Error initializing schema on shard {shard_id}: {e}") 