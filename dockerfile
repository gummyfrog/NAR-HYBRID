FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc make && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
# COPY ./src /app/src
# COPY ./misc/Python /app/misc/Python
COPY . .

RUN chmod +x build.sh && ./build.sh

# Set working directory to where main.py lives
WORKDIR /app/misc/Python

# Run the app
CMD ["python3", "main.py", "--verbose"]
