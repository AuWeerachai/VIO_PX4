from setuptools import setup

package_name = 'vio_px4_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='au',
    maintainer_email='you@example.com',
    description='Bridge Isaac ROS VSLAM odometry to PX4 external vision.',
    license='TBD',
    entry_points={
        'console_scripts': [
            'vio_px4_bridge = vio_px4_bridge.vio_px4_bridge:main',
        ],
    },
)
