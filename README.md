# Numerical Methods in Physics

Repository for laboratory works in the **Numerical Methods in Physics** course.  
It contains implementations of core numerical algorithms used to solve physical problems, along with reports and supporting materials.

## Course overview

The course focuses on numerical approaches that are widely used in physics when analytical solutions are difficult or impossible to obtain.  

The main goal of this repository is to collect:
- laboratory assignments,
- source code implementations,
- reports with results,
- visualizations and supporting materials.

## Repository structure

```text
numerical-methods-in-physics/
├── labs/
│   ├── lab1/
│   ├── lab2/
│   ├── lab3/
│   └── lab4/
├── reports/
│   ├── figures/
│   ├── lab1.md
│   ├── lab2.md
│   ├── lab3.md
│   └── lab4.md
└── README.md
```

### Organization

- `labs/` — source code and materials for each lab.
- `reports/` — reports in Markdown format.
- `figures/` — figures, screenshots, plots, and other media used in reports.

## Labs

### Lab 1
Finite-difference solution of the stationary heat-conduction problem in a cylindrical domain.  
This lab focuses on the formulation, nondimensionalization, grid construction, and derivation of second-order finite-difference equations for the temperature field with mixed boundary conditions.

### Lab 2
Introduction to computational electrodynamics using the 1D FDTD method.  
The lab includes free-space wave propagation, comparison of source implementations, reflection from PML boundaries, reflection from a dielectric interface, transmission through a dielectric slab, and spectral analysis of photonic crystals and Bragg microcavities.

### Lab 3
Introduction to computational fluid dynamics through the stationary advection-diffusion equation.  
This lab uses the finite-volume method on a rectangular grid and studies upwind and TVD discretization schemes, boundary conditions on inlet and outlet boundaries, and the influence of diffusion on the scalar field.

### Lab 4
Steady viscous incompressible flow in a channel.  
This lab focuses on a finite-volume formulation in primitive variables, pressure-velocity coupling, comparison of inlet boundary conditions, analysis of velocity and pressure distributions, and the effect of Reynolds number on the numerical solution.

## Reports

Below is a convenient section for linking reports as they are added to the repository.

- [Lab 1 report](./reports/lab1.md)
- [Lab 2 report](./reports/lab2.md)
- [Lab 3 report](./reports/lab3.md)
- [Lab 4 report](./reports/lab4.md)

## How to use

1. Open the relevant lab folder.
2. Read the task description or report.
3. Run the code examples for the selected laboratory work.
4. Compare the numerical results with the discussion in the report.

## Notes

The repository is being expanded and organized progressively.  
Some folders may be updated with additional code, reports, or figures over time.

## Author

**fnkb108**  
Course repository for *Numerical Methods in Physics*
