import sys
import os
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Add the current directory to the path
sys.path.append(os.getcwd())

# Try to import the Kafka module
try:
    from src.kafka.producer import get_producer
    print("Successfully imported get_producer")
except ImportError as e:
    print(f"Import error: {e}")

# Try to import the module directly
try:
    import src.kafka.producer
    print("Successfully imported src.kafka.producer")
except ImportError as e:
    print(f"Import error: {e}") 