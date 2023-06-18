import unittest

if __name__ == "__main__":
    tests = unittest.TestLoader().discover("./")
    unittest.TextTestRunner().run(tests)
