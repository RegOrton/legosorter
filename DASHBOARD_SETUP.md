# Dashboard Setup Instructions

## Running the Frontend Dashboard

The Next.js frontend dashboard needs to be run in an environment with Node.js installed. There are two options:

### Option 1: Local Development (Windows/Mac/Linux with Node.js)

If you have Node.js installed locally:

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at: http://localhost:3000

### Option 2: Docker Container (Recommended for Production)

Create a Dockerfile for the frontend:

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Expose port
EXPOSE 3000

# Start the app
CMD ["npm", "start"]
```

Then add to `docker-compose.yml`:

```yaml
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    networks:
      - lego-network

  vision:
    # ... existing vision service config
    networks:
      - lego-network

networks:
  lego-network:
    driver: bridge
```

Start everything with:

```bash
docker-compose up --build
```

## Accessing the Dashboard

Once started, navigate to:

- **Training Dashboard**: http://localhost:3000/training
- **Settings Page**: http://localhost:3000/settings

## Features

### Training Dashboard
- View live training progress
- See current settings (dataset, epochs, batch size, camera)
- View triplet images (anchor, positive, negative)
- Monitor training logs in real-time
- Start/Stop training

### Settings Page
- Configure dataset source (LDraw Python, LDView Renders, Rebrickable CGI)
- Set training parameters (epochs, batch size)
- Select camera source (USB, CSI, HTTP)
- Save settings persistently
- Reset to defaults

Settings are stored in `vision/output/settings.json` and persist across container restarts.
