from tensorpy import image_base

result = image_base.classify(
    "https://raw.githubusercontent.com/TensorPy/TensorPy/master/examples/images/happy_animal.jpg")  # noqa
print("\nBest match classification:\n%s\n" % result)
