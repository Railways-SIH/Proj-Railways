const CustomError = require('./custom-error');

class UnauthorizedError extends CustomError {
  constructor(message) {
    super(message, 401);
  }
}

module.exports = UnauthorizedError;
