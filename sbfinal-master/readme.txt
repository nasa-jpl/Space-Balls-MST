Steps to run basic_daily_loop.py:

1. clone the repository

2. Run the following commands:
conda activate monte168
cd <cloned-repo>/sbfinal-master
python -m pip install --no-deps -e .

2. Move all the albedo_and_thermal folder (sent separately) to config/earth/

3. Open the src/SpaceBalls/paths file and set the desired OUTPUT_DIR. MEDIA_DIR is only for pre/postproc results (unrelated to basic_daily_loop.py)

4. Modify the example_input.json input file or create a new one. Outputs will be saved in a folder in OUTPUT_DIR with the same name as the input file
NOTES:
    - The SC shape can be either "sphere" or "sphereN", where N is the number of facets. If N={6, 20, 24, 48, 60, 72, 92 or 96}, tabulated mesh from spherical quadrature grid is used (see get_faceted_sphere_mesh in src/SpaceBalls/sph_meshing). Otherwise, golden spiral mesh is created (see function fibonacci_sphere in src/SpaceBalls/sph_meshing)
    - the field EEI_truth encompasses all the environmental input parameters that affect EEI in one way or another. See src/SpaceBalls/radiation_settings: radiation_settings_from_EEI_truth_name for all existing cases
    - the field n_rings refers to the number of rings to be used for Earth Radiation Pressure integration with the Knocke model. NOTE that any number <40 gives significantly biased results, yet numbers >10 make the daily propagations and time series computation exponentially slower. This can be addressed by using a larger delta_t_out

5. Write down the proper input name in the input field in basic_daily_loop.py and run the script with the active conda environment monte168.