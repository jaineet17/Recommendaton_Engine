#!/usr/bin/env python3
"""
Script to set up Kafka topics for the Amazon Recommendation Engine.
"""

import argparse
import logging
import os
import time
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
from src.data.database import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up Kafka topics for the Amazon Recommendation Engine')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers (comma-separated)')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to the config file')
    parser.add_argument('--replication-factor', type=int, default=1,
                        help='Replication factor for topics')
    parser.add_argument('--partitions', type=int, default=3,
                        help='Number of partitions for topics')
    return parser.parse_args()

def get_topic_list(config):
    """Get the list of topics from the configuration."""
    kafka_config = config.get('kafka', {})
    topics = kafka_config.get('topics', {})
    
    # Extract the topic names
    topic_names = [topic_name for topic_name in topics.values()]
    
    logger.info(f"Found {len(topic_names)} topics in configuration: {', '.join(topic_names)}")
    return topic_names

def create_topics(bootstrap_servers, topics, replication_factor, partitions):
    """Create Kafka topics."""
    admin_client = None
    retry_count = 5
    retry_delay = 5  # seconds
    
    # Try to connect to Kafka with retries
    for attempt in range(retry_count):
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                client_id='amazon-topic-setup'
            )
            break
        except NoBrokersAvailable:
            if attempt < retry_count - 1:
                logger.warning(f"Failed to connect to Kafka brokers. Retrying in {retry_delay} seconds... ({attempt+1}/{retry_count})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Kafka brokers after {retry_count} attempts")
                raise
    
    if not admin_client:
        logger.error("Failed to create Kafka admin client")
        return False
    
    # Create the topics
    topic_list = []
    for topic in topics:
        topic_list.append(NewTopic(
            name=topic,
            num_partitions=partitions,
            replication_factor=replication_factor
        ))
    
    # Create the topics
    created_topics = []
    failed_topics = []
    
    for topic in topic_list:
        try:
            admin_client.create_topics(new_topics=[topic], validate_only=False)
            created_topics.append(topic.name)
            logger.info(f"Created topic: {topic.name}")
        except TopicAlreadyExistsError:
            logger.info(f"Topic already exists: {topic.name}")
            created_topics.append(topic.name)
        except Exception as e:
            logger.error(f"Failed to create topic {topic.name}: {e}")
            failed_topics.append(topic.name)
    
    # Close the admin client
    admin_client.close()
    
    # Log results
    if created_topics:
        logger.info(f"Successfully created or verified {len(created_topics)} topics: {', '.join(created_topics)}")
    
    if failed_topics:
        logger.error(f"Failed to create {len(failed_topics)} topics: {', '.join(failed_topics)}")
        return False
    
    return True

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set the config path from args or environment
    if args.config_path:
        os.environ['CONFIG_PATH'] = args.config_path
    
    # Load the configuration
    config = load_config()
    
    # Get the list of topics
    topics = get_topic_list(config)
    
    # Create the topics
    bootstrap_servers = args.bootstrap_servers.split(',')
    success = create_topics(bootstrap_servers, topics, args.replication_factor, args.partitions)
    
    if success:
        logger.info("Kafka topics setup completed successfully")
        return 0
    else:
        logger.error("Kafka topics setup failed")
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code) 