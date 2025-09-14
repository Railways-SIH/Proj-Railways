const User = require('../models/user');
const jwt = require('jsonwebtoken');
const BadRequest = require('../errors/bad-request');
const UnauthenticatedError = require('../errors/unauthenticated');

const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret_key';

// Signup controller
exports.signup = async (req, res, next) => {
  const { name, email, password } = req.body;
  try {
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return next(new BadRequest('User already exists with this email'));
    }
    // Create new user
    const user = new User({ name, email, password });
    await user.save();
    res.status(201).json({ message: 'User created successfully' });
  } catch (error) {
    next(error);
  }
};

// Signin controller
exports.signin = async (req, res, next) => {
  const { email, password } = req.body;
  try {
    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return next(new BadRequest('Invalid email or password'));
    }
    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return next(new BadRequest('Invalid email or password'));
    }
    // Generate JWT token
    const token = jwt.sign({ userId: user._id }, JWT_SECRET, { expiresIn: '1d' });
    res.status(200).json({ token, user: { id: user._id, name: user.name, email: user.email } });
  } catch (error) {
    next(error);
  }
};
