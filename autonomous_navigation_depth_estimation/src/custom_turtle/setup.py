from setuptools import setup

package_name = 'custom_turtle'

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
    maintainer='bruno',
    maintainer_email='brunoluvizott26@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtle_node = custom_turtle.turtle_node:main',
            'turtle_movement = custom_turtle.turtle_movement:main',
            'turtle_laser = custom_turtle.turtle_laser:main',
        ],
    },
)
