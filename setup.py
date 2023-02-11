from setuptools import setup

package_name = 'ros2_detect_blue'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matt',
    maintainer_email='matthewwilliamlock@gmail.com',
    description='ros2_detect_blue is a ROS2 package that detects blue objects in an image',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_blue = ros2_detect_blue.detect_blue:main'
        ],
    },
)
