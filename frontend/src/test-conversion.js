// Test the API conversion function
import optimizerAPI from './api/optimizerApi.js';

// Sample backend data (same structure as what the API returns)
const sampleBackendTrains = [
  {
    "id": "T1",
    "name": "Express Train",
    "type": "express", 
    "priority": 1,
    "start": "A",
    "end": "D",
    "departure": 0,
    "speed": 60,
    "status": "scheduled",
    "delay": 0,
    "route": ["A", "C", "D"],
    "schedule": {
      "C": { "arrival": 18, "platform": 1 },
      "D": { "arrival": 27, "platform": 1 }
    }
  },
  {
    "id": "T2", 
    "name": "Passenger Train",
    "type": "passenger",
    "priority": 2,
    "start": "A",
    "end": "D", 
    "departure": 1,
    "speed": 50,
    "status": "running",
    "delay": 3,
    "route": ["A", "C", "D"],
    "schedule": {
      "C": { "arrival": 19, "platform": 1 },
      "D": { "arrival": 29, "platform": 1 }
    }
  }
];

console.log('🔧 Testing conversion function...');
console.log('Input (backend format):', sampleBackendTrains);

try {
  const convertedTrains = optimizerAPI.convertTrainDataToFrontend(sampleBackendTrains);
  console.log('✅ Output (frontend format):', convertedTrains);
  
  // Check specific fields
  convertedTrains.forEach(train => {
    console.log(`Train ${train.id}:`);
    console.log(`  - Name: ${train.name}`);
    console.log(`  - Section: ${train.section}`);
    console.log(`  - Route: ${train.route}`);
    console.log(`  - Status: ${train.status} (${train.statusType})`);
  });
  
} catch (error) {
  console.error('❌ Conversion failed:', error);
}