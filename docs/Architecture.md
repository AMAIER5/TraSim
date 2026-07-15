Architecture Specification
Koppelgetriebe Simulator
Version: 0.1.0
Status: Draft
Language: English (code/documentation identifiers)
1. Purpose
The project provides a simulation framework for planar and spatial linkage mechanisms.
The primary goals are:
reliable kinematic simulation,
support for arbitrary numbers of linkage stages,
extension towards optimization algorithms,
reproducible engineering calculations.
The simulator shall support:
3D lever axes,
spherical joints,
rigid coupling rods,
multi-stage mechanisms,
evolutionary optimization.
2. Coordinate System
The simulator uses a right-handed Cartesian coordinate system.
             +Z
             |
             |
             |
             o──────── +X
            /
           /
         +Y
Coordinates: Point3D(x, y, z)
Unit: millimeter [mm]
3. Units
User interface
Lengths:
mm
Angles:
deg
Internal calculations
Lengths:
mm
Angles:
rad
All conversion between degrees and radians happens at the interface boundary.
No internal object shall store angles in degrees.
4. Rotation Convention
Rotations follow the right-hand rule internally.
A positive mathematical rotation is:
counter-clockwise
when looking along the positive rotation axis toward the origin.
For engineering input/output:
angle_deg
may use a mechanism-specific convention.
Conversion shall happen only in the input/output layer.
5. Basic Geometric Entities
Point3D
Represents a position in space.
Properties:
x
y
z
Example:
Point3D(10.0, 20.0, 5.0)
Vector3D
Represents a direction or displacement.
Properties:
x
y
z
Operations:
addition
subtraction
scaling
dot product
cross product
normalization
angle calculation
rotation
Rotation
A rotation is represented internally by:
Quaternion
Advantages:
no gimbal lock,
stable interpolation,
efficient composition,
suitable for animation.
6. Lever Definition
A lever is defined by:
Lever
│
├── pivot_point : Point3D
├── axis        : Vector3D
├── length      : float
├── reference   : Vector3D
└── angle       : float
The end point is calculated as:
end = pivot + rotated(reference * length)
7. Coupler Rod Definition
A rod is an ideal rigid body.
Properties:
Rod
│
├── point_a
├── point_b
└── length
Constraints:
distance(point_a, point_b) = length
The rod ends are ideal spherical joints.
8. Mechanism Structure
A mechanism consists of stages:
Mechanism

    Stage 1
       |
       |
    Stage 2
       |
       |
    Stage 3
Each stage contains:
input lever

output lever

coupler rod
All pivots are fixed to the global frame.
9. Solver Requirements
The solver shall:
accept an initial input angle,
simulate in positive and negative directions,
stop at configured limits,
stop when no valid solution exists,
return NaN for impossible positions,
preserve continuous angle tracking.
10. Numerical Rules
All geometric comparisons use tolerances.
Example:
distance_error < tolerance
Never compare floating point values directly.
Forbidden:
if value == 0:
Required:
if abs(value) < tolerance:
11. Testing Philosophy
Every mathematical module requires:
unit tests,
boundary tests,
invalid input tests,
numerical stability tests.
No mechanism module may depend on untested geometry functions.
12. Future Optimization Interface
The solver shall provide a deterministic interface:
Input:
mechanism parameters
simulation range
Output:
positions
angles
validity information
The optimizer shall not access internal solver details.
millimeter [mm]
