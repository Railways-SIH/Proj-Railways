require('dotenv').config();
const connectDB = require('./config/connect');


const express = require('express');
const app = express();

const authRoutes = require('./routes/authRoutes');

app.use(express.json());

// Use auth routes
app.use('/api/auth', authRoutes);

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
