# Change log


## NNDT 0.0.2 (16 Oct 2022)
* **The partial becomes complete; the crooked, straight; the empty, full; the worn out, new.**
* Space model is fully rewritten
* Visualization of tree and 3D object via `.plot()` and `.print()`
* Short and easy import
* Conversion of signed distance tensor to signed distance function using MLP training
* 30 methods are available in the second version of the space model:
  * Available as functions (12)
    * Space model loaders (6): `load_from_path`, `load_from_file_lists`, `load_only_one_file`, `load_txt`, `load_sdt`, `load_mesh_obj`
    * Space model serialization (2): `to_json`, `save_space_to_file`
    * Space model deserialization (2): `read_space_from_file`, `from_json`
    * Primitive generation (1): `add_sphere`
    * Space model modification (1): `split_node_test_train`'
  * Available as methods on the space model tree (18)
    * General (3): `.plot`, `.print`, `.unload_from_memory`
    * Sampling (4): `.sampling_eachN_from_mesh`, `.sampling_grid`, `.sampling_grid_with_noise`, `.sampling_uniform`
    * Surface data (6): `.surface_ind2rgba`, `.surface_ind2xyz`, `.surface_xyz2ind`, `.surface_xyz2localsdt`, `.surface_xyz2rgba`, `.surface_xyz2sdt`
    * Transformations (4): `.transform_sdt_ns2ps`, `.transform_sdt_ps2ns`, `.transform_xyz_ns2ps`, `.transform_xyz_ps2ns`
    * Save (1): `.save_mesh`
* One real demo (still on space model of the first version)
  * Shape interpolation for building of Computation anatomy atlas
* Code quality and team
  * CI pipeline for new pull requests
  * 73% of code coverage in CodeCov
  * `A` grade, 10 issues is  CodeFactor
  * 7 contributors
* Known issues:
  * Wiki page is outdated
  * Lack of documentation
  * Bugs with SDT requests and cube marching
  * Some test is disabled for CI pipeline restrictions

## NNDT 0.0.1 (16 Sep 2022)
* **A journey of a thousand miles begins with a single step.**
* The first release of the code with all key features.
* [Wiki page](https://github.com/KonstantinUshenin/nndt/wiki) for maintainers and contributors.
* Notebook tutorials:
    * Space models
    * Trainable tasks
* Demos:
    * Mesh segmentation
    * Shape regression
    * Eikonal in 3D model
