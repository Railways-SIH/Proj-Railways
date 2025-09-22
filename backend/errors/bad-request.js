const CustomError = require('./custom-error');

class BadRequestError extends CustomError {
  constructor(message) {
    super(message, 400);
  }
}

module.exports = BadRequestError;
