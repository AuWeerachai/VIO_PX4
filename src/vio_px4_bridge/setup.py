from setuptools import setup

package_name = 'vio_px4_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_data={package_name: ['data/*.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='au',
    maintainer_email='you@example.com',
    description='Bridge Isaac ROS VSLAM odometry to PX4 (EV and internship-style GPS).',
    license='TBD',
    entry_points={
        'console_scripts': [
            'vio_px4_gps_bridge = vio_px4_bridge.vio_px4_gps_bridge:main',
            'mavros_ev_bridge = vio_px4_bridge.mavros_ev_bridge:main',
            'cuvslam_body_relay = vio_px4_bridge.cuvslam_body_relay:main',
        ],
    },
)
