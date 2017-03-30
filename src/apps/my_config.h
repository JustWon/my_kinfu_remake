
namespace MY_CONFIG{
	const int device_num = 0; // it can be changed

	const int max_view_num = 2;

	const int vh_resize_factor = 2; // size of the generated depth image
	const int vh_dividing_factor = 5;

	const int dividing_factor_for_seq_tof_img = 1;
	const int multiplying_factor_for_save_depth_image = 1000;

	const bool seq_direct_saving = true;
	const bool color_overlap = true;
	const bool use_all_frames = true;

	const double tsdf_min_camera_movement = 0.0f;
	const double raycast_step_factor = 0.1f;

	// 20161204_231541: person
	// 20161204_232513: panel 1
	// 20161204_232836: panel 2
	// 20161204_233533: Objects

	// 20170113_142649: SH
	// 20170113_143623: MS
	// 20170114_001102: DW
	const std::string seq_id = "20170114_001102";
}
