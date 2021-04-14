#include "createRDeltaT.H"

Info<< "Reading field p_rgh\n" << endl;
volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"


Info<< "Reading transportProperties\n" << endl;
immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);

volScalarField& alpha1(mixture.alpha1());
volScalarField& alpha2(mixture.alpha2());

const dimensionedScalar& rho1 = mixture.rho1();
const dimensionedScalar& rho2 = mixture.rho2();


// Need to store rho for ddt(rho, U)
volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    alpha1*rho1 + alpha2*rho2
);
rho.oldTime();


// Mass flux
surfaceScalarField rhoPhi
(
    IOobject
    (
        "rhoPhi",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    fvc::interpolate(rho)*phi
);

typedef incompressibleInterPhaseTransportModel
    <
        immiscibleIncompressibleTwoPhaseMixture
    > transportModelType;

autoPtr<transportModelType> turbulence
(
    new transportModelType
    (
        rho, U, phi, rhoPhi, mixture
    )
);

#include "readGravitationalAcceleration.H"
#include "readhRef.H"
#include "gh.H"


volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    p_rgh + rho*gh
);

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell
(
    p,
    p_rgh,
    pimple.dict(),
    pRefCell,
    pRefValue
);

if (p_rgh.needReference())
{
    p += dimensionedScalar
    (
        "p",
        p.dimensions(),
        pRefValue - getRefCellValue(p, pRefCell)
    );
    p_rgh = p - rho*gh;
}

mesh.setFluxRequired(p_rgh.name());
mesh.setFluxRequired(alpha1.name());

// MULES compressed flux is registered in case scalarTransport FO needs it.
surfaceScalarField alphaPhiUn
(
    IOobject
    (
        "alphaPhiUn",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(phi.dimensions(), Zero)
);

// Linearly independent tensors T0, ..., T9 made from
// different combinations of strain rate and rotation rate tensors.
volSymmTensorField strainRateTensor
(
    IOobject
    (
        "strainRateTensor",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    symm(fvc::grad(U))
);

volTensorField rotationRateTensor
(
    IOobject
    (
        "rotationRateTensor",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    skew(fvc::grad(U))
);

volTensorField T0
(
    IOobject
    (
        "T0",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    tensor::I & strainRateTensor
);

volTensorField T1
(
    IOobject
    (
        "T1",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (strainRateTensor & rotationRateTensor) - (rotationRateTensor & strainRateTensor)
);

volTensorField T2
(
    IOobject
    (
        "T2",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (strainRateTensor & strainRateTensor) - (tensor::I * (tr(strainRateTensor & strainRateTensor) / 3))
);

volTensorField T3
(
    IOobject
    (
        "T3",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (rotationRateTensor & rotationRateTensor) - (tensor::I * (tr(rotationRateTensor & rotationRateTensor) / 3))
);

volTensorField T4
(
    IOobject
    (
        "T4",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (rotationRateTensor & strainRateTensor & strainRateTensor) -
		(strainRateTensor & strainRateTensor & rotationRateTensor)
);

volTensorField T5
(
    IOobject
    (
        "T5",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (rotationRateTensor & rotationRateTensor & strainRateTensor) +
		(strainRateTensor & rotationRateTensor & rotationRateTensor) -
		(tensor::I * (2 * tr(strainRateTensor & rotationRateTensor & rotationRateTensor) / 3))
);

volTensorField T6
(
    IOobject
    (
        "T6",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (rotationRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) -
		(rotationRateTensor & rotationRateTensor & strainRateTensor & rotationRateTensor)
);

volTensorField T7
(
    IOobject
    (
        "T7",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (strainRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor) -
		(strainRateTensor & strainRateTensor & rotationRateTensor & strainRateTensor)
);

volTensorField T8
(
    IOobject
    (
        "T8",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (rotationRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor) +
		(strainRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) -
		(tensor::I * (2 * tr(strainRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) / 3))
);

volTensorField T9
(
    IOobject
    (
        "T9",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    (rotationRateTensor & strainRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) -
		(rotationRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor & rotationRateTensor)
);

volScalarField I0
(
    IOobject
    (
        "I0",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    tr(strainRateTensor & strainRateTensor)
);

volScalarField I1
(
    IOobject
    (
        "I1",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    tr(rotationRateTensor & rotationRateTensor)
);

volScalarField I2
(
    IOobject
    (
        "I2",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    tr(strainRateTensor & strainRateTensor & strainRateTensor)
);

volScalarField I3
(
    IOobject
    (
        "I3",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    tr(rotationRateTensor & rotationRateTensor & strainRateTensor)
);

volScalarField I4
(
    IOobject
    (
        "I4",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    tr(rotationRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor)
);

#include "createMRF.H"
#include "createFvOptions.H"
