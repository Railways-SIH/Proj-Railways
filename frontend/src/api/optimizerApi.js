// optimizerApi.js - Frontend API service for connecting to backend optimizer
const API_BASE_URL = 'http://localhost:5000/api/optimizer';

class OptimizerAPI {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Fetch network topology and structure
  async getNetworkData() {
    try {
      const response = await fetch(`${this.baseURL}/network`);
      if (!response.ok) {
        throw new Error(`Network request failed: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching network data:', error);
      throw error;
    }
  }

  // Fetch train schedules and optimization results
  async getScheduleData() {
    try {
      const response = await fetch(`${this.baseURL}/schedule`);
      if (!response.ok) {
        throw new Error(`Schedule request failed: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching schedule data:', error);
      throw error;
    }
  }

  // Run optimization with custom parameters
  async optimizeSchedule(trains, constraints = {}) {
    try {
      const response = await fetch(`${this.baseURL}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ trains, constraints }),
      });
      
      if (!response.ok) {
        throw new Error(`Optimization request failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error running optimization:', error);
      throw error;
    }
  }

  // Get real-time train status updates
  async getTrainStatus(trainId = null) {
    try {
      const url = trainId 
        ? `${this.baseURL}/status/${trainId}` 
        : `${this.baseURL}/status`;
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Status request failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching train status:', error);
      throw error;
    }
  }

  // Convert backend data to frontend format
  convertTrainDataToFrontend(backendTrains) {
    return backendTrains.map(train => ({
      id: train.id,
      name: train.name || `Train ${train.id}`,
      number: this.generateTrainNumber(train.type, train.id),
      type: train.type,
      priority: train.priority,
      section: this.mapStationToSection(train.start),
      speed: train.speed,
      destination: this.getDestinationName(train.end),
      status: this.mapStatus(train.status),
      delay: train.delay || 0,
      route: this.mapRouteToSections(train.route),
      statusType: this.getStatusType(train.status),
      departure: train.departure,
      schedule: train.schedule || {},
      platforms: this.extractPlatformInfo(train.schedule)
    }));
  }

  // Convert backend network data to frontend format
  convertNetworkDataToFrontend(networkData) {
    return {
      stations: networkData.stations?.map(station => ({
        id: station.id,
        name: station.name,
        coordinates: station.coordinates || this.getStationCoordinates(station.id),
        platforms: station.platforms || 2,
        status: 'operational'
      })) || [],
      
      tracks: networkData.tracks?.map(track => ({
        id: `${track.from}-${track.to}`,
        from: track.from,
        to: track.to,
        length: track.length,
        speedLimit: track.speedLimit,
        status: track.status || 'free',
        occupiedBy: track.occupiedBy || null
      })) || []
    };
  }

  // Helper methods
  generateTrainNumber(type, id) {
    const prefixes = {
      'express': '123',
      'passenger': '456', 
      'freight': '789'
    };
    const prefix = prefixes[type] || '000';
    const number = id.replace('T', '');
    return `${prefix}0${number}`;
  }

  mapStationToSection(station) {
    const stationToSection = {
      'A': '1R',
      'B': '3L', 
      'C': '5L',
      'D': '7L'
    };
    return stationToSection[station] || '1R';
  }

  getDestinationName(station) {
    const stationNames = {
      'A': 'New Delhi Junction',
      'B': 'Mumbai Central',
      'C': 'Kolkata Howrah', 
      'D': 'Chennai Central'
    };
    return stationNames[station] || `Station ${station}`;
  }

  mapStatus(status) {
    const statusMap = {
      'scheduled': 'Scheduled',
      'running': 'Running',
      'waiting': 'Waiting',
      'delayed': 'Delayed',
      'stopped': 'Stopped'
    };
    return statusMap[status] || 'Unknown';
  }

  getStatusType(status) {
    if (status === 'delayed') return 'delayed';
    if (status === 'stopped' || status === 'waiting') return 'stopped';
    return 'running';
  }

  mapRouteToSections(route) {
    if (!route) return [];
    return route.map(station => this.mapStationToSection(station));
  }

  extractPlatformInfo(schedule) {
    if (!schedule) return {};
    const platforms = {};
    Object.entries(schedule).forEach(([station, info]) => {
      if (info.platform) {
        platforms[station] = info.platform;
      }
    });
    return platforms;
  }

  getStationCoordinates(stationId) {
    const coordinates = {
      'A': { x: 100, y: 200 },
      'B': { x: 300, y: 200 },
      'C': { x: 500, y: 200 },
      'D': { x: 700, y: 200 }
    };
    return coordinates[stationId] || { x: 0, y: 0 };
  }
}

// Export singleton instance
const optimizerAPI = new OptimizerAPI();
export default optimizerAPI;