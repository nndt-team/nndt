# Change log


## NNDT 0.0.2 (16 Oct 2022)
* **The partial becomes complete; the crooked, straight; the empty, full; the worn out, new.**
* *Now we have 7 contributors. Thank you all for new docs, features, issues and CI/CD pipeline!*
* The space model (`space`) is pandas for 3D objects. It is completely redesigned and is now available as `space2`. The previous version of `space` is still here, but will be removed soon
* Visualization of tree and 3D objects via `.plot()` and `.print()`
* New short and easy import for `space2`
* 30 methods are available in the second version of the space model
  * Available as functions (12)
    * Space model loaders (6): `load_from_path`, `load_from_file_lists`, `load_only_one_file`, `load_txt`, `load_sdt`, `load_mesh_obj`
    * Space model serialization (2): `to_json`, `save_space_to_file`
    * Space model deserialization (2): `read_space_from_file`, `from_json`
    * Primitive generation (1): `add_sphere`
    * Space model modification (1): `split_node_test_train`'
  * Available as methods in the space model tree (18)
    * General (3): `.plot`, `.print`, `.unload_from_memory`
    * Sampling (4): `.sampling_eachN_from_mesh`, `.sampling_grid`, `.sampling_grid_with_noise`, `.sampling_uniform`
    * Surface data (6): `.surface_ind2rgba`, `.surface_ind2xyz`, `.surface_xyz2ind`, `.surface_xyz2localsdt`, `.surface_xyz2rgba`, `.surface_xyz2sdt`
    * Transformations (4): `.transform_sdt_ns2ps`, `.transform_sdt_ps2ns`, `.transform_xyz_ns2ps`, `.transform_xyz_ps2ns`
    * Save (1): `.save_mesh`
* New modules for Haiku: MLP with Lipschitz regularization (LipMLP, LipLinear)
* Functions for iteration over N-simplex with barycentric coordinates in `math_core.py`
* Serialization and deserialization of the `space2` model
* Conversion of signed distance tensor (SDT) to signed distance function (SDF) using MLP training (still uses the previous version of space model)
* One real demo for a research paper (still uses the previous version of space model)
  * Shape interpolation for building a computational anatomy atlas
* Draft documentation for some methods
* Code quality
  * CI pipeline for new pull requests
  * 74% of code coverage in CodeCov
  * `A` grade, 10 issues in CodeFactor
* Known issues
  * Wiki page is outdated
  * Lack of documentation
  * Bugs with SDT requests and cube marching
  * Some test is disabled because of CI pipeline restrictions

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
