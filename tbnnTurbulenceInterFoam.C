/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2020 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    tbnnTurbulenceInterFoam

Group
    grpMultiphaseSolvers

Description
    Solver for two incompressible, isothermal immiscible fluids using a VOF
    (volume of fluid) phase-fraction based interface capturing approach,
    with optional mesh motion and mesh topology changes including adaptive
    re-meshing.

\*---------------------------------------------------------------------------*/
#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "CMULES.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "incompressibleInterPhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "CorrectPhi.H"
#include "fvcSmooth.H"

#include "tensorflow/c/c_api.h"
#include "addNN.h"
#include "NNInfoParser.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Solver for two incompressible, isothermal immiscible fluids"
        " using VOF phase-fraction based interface capturing.\n"
        "With optional mesh motion and mesh topology changes including"
        " adaptive re-meshing."
    );


    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createAlphaFluxes.H"
    #include "initCorrectPhi.H"
    #include "createUfIfPresent.H"

    #include "initNN.H"

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    Info<< "\nStarting time loop\n" << endl;
    while (runTime.run())
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "alphaCourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                mesh.update();

                if (mesh.changing())
                {
                    // Do not apply previous time-step mesh compression flux
                    // if the mesh topology changed
                    if (mesh.topoChanging())
                    {
                        talphaPhi1Corr0.clear();
                    }

                    gh = (g & mesh.C()) - ghRef;
                    ghf = (g & mesh.Cf()) - ghRef;

                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

                        #include "correctPhi.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);

                        mixture.correct();
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "alphaControls.H"
            #include "alphaEqnSubCycle.H"

            mixture.correct();

            if (pimple.frozenFlow())
            {
                continue;
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

		// Calc T0, ..., T9
		gradU = fvc::grad(U);
		inv1GradU = tr(gradU);
		inv2GradU = tr(gradU) * tr(gradU) - tr(gradU & gradU);
		strainRateTensor = symm(gradU);
		rotationRateTensor = skew(gradU);
    	T0 = tensor::I & strainRateTensor;
    	T1 = (strainRateTensor & rotationRateTensor) - (rotationRateTensor & strainRateTensor);
    	T2 = (strainRateTensor & strainRateTensor) - (tensor::one * (tr(strainRateTensor & strainRateTensor) / 3));
    	T3 = (rotationRateTensor & rotationRateTensor) -
			(tensor::one * (tr(rotationRateTensor & rotationRateTensor) / 3));
    	T4 = (rotationRateTensor & strainRateTensor & strainRateTensor) -
			(strainRateTensor & strainRateTensor & rotationRateTensor);
    	T5 = (rotationRateTensor & rotationRateTensor & strainRateTensor) +
			(strainRateTensor & rotationRateTensor & rotationRateTensor) -
			(tensor::I * (2 * tr(strainRateTensor & rotationRateTensor & rotationRateTensor) / 3));
    	T6 = (rotationRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) -
			(rotationRateTensor & rotationRateTensor & strainRateTensor & rotationRateTensor);
    	T7 = (strainRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor) -
			(strainRateTensor & strainRateTensor & rotationRateTensor & strainRateTensor);
    	T8 = (rotationRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor) +
			(strainRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) -
			(tensor::I * (2 * tr(strainRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) / 3));
    	T9 = (rotationRateTensor & strainRateTensor & strainRateTensor & rotationRateTensor & rotationRateTensor) -
			(rotationRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor & rotationRateTensor);
    	I0 = tr(strainRateTensor & strainRateTensor);
    	I1 = tr(rotationRateTensor & rotationRateTensor);
    	I2 = tr(strainRateTensor & strainRateTensor & strainRateTensor);
    	I3 = tr(rotationRateTensor & rotationRateTensor & strainRateTensor);
    	I4 = tr(rotationRateTensor & rotationRateTensor & strainRateTensor & strainRateTensor);
		gradP = fvc::grad(p_rgh);
		magGradP = mag(gradP);
		gradAW = fvc::grad(alpha1);
		magGradAW = mag(gradAW);

        if (correct_fields_with_nn) 
        {
            try
            {
                Info << "Starting fields correction witn NN\n" << endl;
                #include "prepareData.H"

                std::vector<double> output_data(nn_output.branch_size);
                for (size_t i = 0; i < U.size(); ++i)
                {
                    nn.Run(
                        output_data, input_data[i], inputs_names
                    );

                    //TODO: correct processing of NN outputs from the config 

                    //temporary solution
                    U_corr.data()[i] = {output_data[0], output_data[1], output_data[2]};
                    alpha1_corr.data()[i] = output_data[3];
                    p_rgh_corr.data()[i] = output_data[4];
                }

                U = U + U_corr;
                alpha1 = alpha1 + alpha1_corr;
                p_rgh = p_rgh + p_rgh_corr;

                Info << "End of fields correction witn NN\n" << endl;
            }
            catch (const std::exception &e)
            {
                Info << "Failed to correct fields with NN: "
                << e.what() << endl;
            }
            catch (...)
            {
                Info << "Failed to correct fields with NN." << endl;
            }
        }


        runTime.write();
        runTime.printExecutionTime(Info);

        }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
