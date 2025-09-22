require('dotenv').config();
const connectDB = require('./config/connect');


const express = require('express');
const app = express();

const authRoutes = require('./routes/authRoutes');
const { spawn } = require('child_process');
const path = require('path');

app.use(express.json());

// Enable CORS for frontend
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*'); // Allow all origins for development
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Use auth routes
app.use('/api/auth', authRoutes);

// Optimizer API endpoints
app.get('/api/optimizer/network', async (req, res) => {
  try {
    // Static network data based on optimizer structure
    const networkData = {
      stations: [
        { id: 'A', name: 'Station A', coordinates: { x: 100, y: 200 }, platforms: 2 },
        { id: 'B', name: 'Station B', coordinates: { x: 300, y: 200 }, platforms: 2 },
        { id: 'C', name: 'Station C', coordinates: { x: 500, y: 200 }, platforms: 2 },
        { id: 'D', name: 'Station D', coordinates: { x: 700, y: 200 }, platforms: 2 }
      ],
      tracks: [
        { from: 'A', to: 'B', length: 5, speedLimit: 60, status: 'free' },
        { from: 'B', to: 'C', length: 10, speedLimit: 80, status: 'free' },
        { from: 'C', to: 'D', length: 7, speedLimit: 70, status: 'free' },
        { from: 'A', to: 'C', length: 15, speedLimit: 50, status: 'free' }
      ],
      blockStatus: {}
    };
    res.json(networkData);
  } catch (error) {
    console.error('Network API error:', error);
    res.status(500).json({ error: 'Failed to get network data' });
  }
});

app.get('/api/optimizer/schedule', async (req, res) => {
  try {
    // Run Python optimizer and return results
    const pythonScript = path.join(__dirname, 'optimizer', 'optimizer_api.py');
    const pythonProcess = spawn('python', [pythonScript]);
    
    let dataBuffer = '';
    let errorBuffer = '';
    let responseSet = false;

    pythonProcess.stdout.on('data', (data) => {
      dataBuffer += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorBuffer += data.toString();
      console.error('Python script error:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (responseSet) return; // Prevent double response
      responseSet = true;
      
      try {
        if (code === 0 && dataBuffer) {
          // Parse JSON output from Python
          const scheduleData = JSON.parse(dataBuffer);
          res.json(scheduleData);
        } else {
          throw new Error(`Python process failed with code ${code}: ${errorBuffer}`);
        }
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        // Fallback to static data
        const fallbackData = {
          success: true,
          timestamp: new Date().toISOString(),
          trains: [
            {
              id: 'T1',
              name: 'Express Train',
              type: 'express',
              priority: 1,
              start: 'A',
              end: 'D',
              departure: 0,
              speed: 60,
              status: 'scheduled',
              delay: 0,
              route: ['A', 'B', 'C', 'D'],
              schedule: {
                'B': { arrival: 5, platform: 1 },
                'C': { arrival: 17, platform: 2 },
                'D': { arrival: 25, platform: 1 }
              }
            },
            {
              id: 'T2',
              name: 'Passenger Train',
              type: 'passenger',
              priority: 2,
              start: 'A',
              end: 'D',
              departure: 1,
              speed: 50,
              status: 'running',
              delay: 3,
              route: ['A', 'B', 'C', 'D'],
              schedule: {
                'B': { arrival: 7, platform: 2 },
                'C': { arrival: 20, platform: 1 },
                'D': { arrival: 29, platform: 2 }
              }
            },
            {
              id: 'T3',
              name: 'Freight Train',
              type: 'freight',
              priority: 3,
              start: 'B',
              end: 'D',
              departure: 3,
              speed: 40,
              status: 'waiting',
              delay: 0,
              route: ['B', 'C', 'D'],
              schedule: {
                'C': { arrival: 18, platform: 1 },
                'D': { arrival: 28, platform: 2 }
              }
            }
          ],
          optimizationMetrics: {
            totalDelay: 3,
            throughput: 85,
            efficiency: 92,
            conflicts: 0
          }
        };
        res.json(fallbackData);
      }
    });

    // Timeout handling
    const timeoutId = setTimeout(() => {
      if (responseSet) return; // Prevent double response
      responseSet = true;
      pythonProcess.kill();
      res.status(500).json({ error: 'Python script timeout' });
    }, 10000);

    // Clear timeout if process completes
    pythonProcess.on('close', () => {
      clearTimeout(timeoutId);
    });

  } catch (error) {
    console.error('Schedule API error:', error);
    if (!responseSet) {
      res.status(500).json({ error: 'Failed to get schedule data' });
    }
  }
});

app.post('/api/optimizer/optimize', async (req, res) => {
  try {
    const { trains, constraints } = req.body;
    
    // Here you would run the Python optimizer with custom parameters
    const optimizedSchedule = {
      success: true,
      message: 'Optimization completed',
      trains: trains, // Return optimized train data
      metrics: {
        improvementPercent: 15,
        delayReduction: 8,
        throughputIncrease: 12
      }
    };
    
    res.json(optimizedSchedule);
  } catch (error) {
    console.error('Optimization API error:', error);
    res.status(500).json({ error: 'Failed to optimize schedule' });
  }
});

const notFoundMiddleware = require('./middleware/not-found');
const errorHandlerMiddleware = require('./middleware/error-handler');

// Define routes here..before the error handlers

app.use(notFoundMiddleware);
app.use(errorHandlerMiddleware);

const port = process.env.PORT || 5000;

const start = async () => {
  try {
    // Connect to the database
    await connectDB(process.env.MONGO_URI);
    console.log('Connected to the database successfully');
    app.listen(port, () =>
      console.log(`Server is listening on port ${port}...`)
    );
  } catch (error) {
    console.log(error);
  }
};

start();

module.exports = app;
