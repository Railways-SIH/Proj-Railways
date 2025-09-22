const { CustomError } = require('../errors')

const errorHandler = (err, req, res, next) => {
  if (err instanceof CustomError) {
    return res.status(err.statusCode).json({
      status: 'error',
      message: err.message,
    })
  }

  console.error('Error:', err)

  return res.status(500).json({
    status: 'error',
    message: 'Internal server error',
  })
}

module.exports = errorHandler
