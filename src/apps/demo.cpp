#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>
#include <stdio.h>
#include <string>
#include "my_config.h"
#include <opencv2/opencv.hpp>

using namespace kfusion;

struct KinFuApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.take_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.iteractive_mode_ = !kinfu.iteractive_mode_;

        if(event.code == '[')
        {
        	if (kinfu.blending_alpha > 0)
        		kinfu.blending_alpha -= 0.1;
        }

        if(event.code == ']')
        {
        	if (kinfu.blending_alpha < 1)
        		kinfu.blending_alpha += 0.1;
        }

        if(event.code == 'a')
        {
        	kinfu.theta_X += 0.01;
        }
        if(event.code == 'z')
		{
			kinfu.theta_X -= 0.01;
		}
        if(event.code == 's')
        {
        	kinfu.theta_Y += 0.01;
        }
        if(event.code == 'x')
		{
			kinfu.theta_Y -= 0.01;
		}
        if(event.code == 'd')
        {
        	kinfu.theta_Z += 0.01;
        }
        if(event.code == 'c')
		{
			kinfu.theta_Z -= 0.01;
		}
    }

    KinFuApp(OpenNISource& source) : exit_ (false),  iteractive_mode_(false), capture_ (source), pause_(false)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("color", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }
    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
    }

    template <typename T>
    std::string to_string(T value)
    {
    	std::ostringstream os ;
    	os << value ;
    	return os.str() ;
    }

    void diplay_generated_volume(KinFu& kinfu, Affine3f cur_view, int view_id, const int mode, int win_xpos, int win_ypos)
    {
    	kinfu.renderImage(view_device_, cur_view, mode);
    	view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
    	view_device_.download(view_host_.ptr<void>(), view_host_.step);
		cv::Mat resized(view_host_.rows/MY_CONFIG::vh_resize_factor, view_host_.cols/MY_CONFIG::vh_resize_factor,CV_8UC4);

		for (int i = 0 ; i < view_host_.rows/MY_CONFIG::vh_resize_factor ; i++)
		for (int j = 0 ; j < view_host_.cols/MY_CONFIG::vh_resize_factor ; j++)
		{
			resized.at<cv::Vec4b>(i,j)[0] = view_host_.at<cv::Vec4b>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j)[0];
			resized.at<cv::Vec4b>(i,j)[1] = view_host_.at<cv::Vec4b>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j)[1];
			resized.at<cv::Vec4b>(i,j)[2] = view_host_.at<cv::Vec4b>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j)[2];
			resized.at<cv::Vec4b>(i,j)[3] = view_host_.at<cv::Vec4b>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j)[3];
		}

		std::string window_name = "view"+ to_string(view_id);
		cv::imshow(window_name, resized);
//		cv::moveWindow(window_name,win_xpos,win_ypos);
    }
    void color_seq_filename(std::string seq_id, int view_count, int seq_count)
    {
//    	if (seq_id == "20161204_232513")
//    		color_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com1/20161204_232513/sequence/color"+to_string(view_count)+"_"+ to_string(seq_count)+".bmp";
//    	else if (seq_id == "20161204_231541")
//			color_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com1/20161204_231541/sequence/color"+to_string(view_count)+"_"+ to_string(seq_count)+".bmp";
//    	else if (seq_id == "20161204_232836")
//    		color_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com1/20161204_232836/sequence/color"+to_string(view_count)+"_"+ to_string(seq_count)+".bmp";
//    	else if (seq_id == "20161204_233533")
//    		color_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com1/20161204_233533/sequence/color"+to_string(view_count)+"_"+ to_string(seq_count)+".bmp";


		color_file_name = "/home/dongwonshin/Documents/Datasets/Sequence Data/20170113 Experiment/server/" + seq_id + "/still/cam" + to_string(view_count) +
				"/color"+ to_string(view_count)+"_"+ to_string(seq_count) + ".bmp" ;

//		std::cout << color_file_name << std::endl;
    }
    void depth_seq_filename(std::string seq_id, int view_count, int seq_count)
    {
    	// for previous exp sequences
    	// for sync between depth cams
//    	if (view_count == 1)
//    		seq_count += 1;
//
//    	if (seq_id == "20161204_232513")
//    		depth_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com"+ to_string(view_count) +"/20161204_232513/sequence/kinect_depth"+to_string(seq_count)+".png";
//    	else if (seq_id == "20161204_231541")
//    		depth_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com"+ to_string(view_count) +"/20161204_231541/sequence/kinect_depth"+to_string(seq_count)+".png";
//    	else if (seq_id == "20161204_232836")
//			depth_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com"+ to_string(view_count) +"/20161204_232836/sequence/kinect_depth"+to_string(seq_count)+".png";
//    	else if (seq_id == "20161204_233533")
//    		depth_file_name = "/home/dongwonshin/Desktop/Sequence Data/20161205 test/com"+ to_string(view_count) +"/20161204_233533/sequence/kinect_depth"+to_string(seq_count)+".png";

		if (view_count == 1)
			depth_file_name = "/home/dongwonshin/Documents/Datasets/Sequence Data/20170113 Experiment/server/" + seq_id + "/still/tof1/kinect_depth1_" + to_string(seq_count) + ".png" ;
		else if (view_count == 2)
			depth_file_name = "/home/dongwonshin/Documents/Datasets/Sequence Data/20170113 Experiment/client/" + seq_id + "/still/tof2/kinect_depth2_" + to_string(seq_count) + ".png" ;

//		std::cout << depth_file_name << std::endl;
    }

    void display_generated_depth(KinFu& kinfu, Affine3f cur_view, int view_id, const int mode, int win_xpos, int win_ypos, int view_count, int seq_count, std::string seq_id)
    {
    	kinfu.renderImage(view_device_, cur_view, mode);
    	view_host_.create(view_device_.rows(), view_device_.cols(), CV_32FC1);
    	view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_/ MY_CONFIG::vh_dividing_factor;
		cv::Mat resized(view_host_.rows/MY_CONFIG::vh_resize_factor, view_host_.cols/MY_CONFIG::vh_resize_factor,CV_32FC1);

		if (MY_CONFIG::color_overlap)
		{
			//  color overlap
			color_seq_filename(seq_id, view_count, seq_count);
			cv::Mat here_img = cv::imread(color_file_name);

			float alpha = blending_alpha;
			for (int i = 0 ; i < view_host_.rows/MY_CONFIG::vh_resize_factor ; i++)
			for (int j = 0 ; j < view_host_.cols/MY_CONFIG::vh_resize_factor ; j++)
			{
				float temp_v=0;
				if (!here_img.empty())
				{
					cv::Vec3b vec_val = here_img.at<cv::Vec3b>(MY_CONFIG::vh_resize_factor*i, MY_CONFIG::vh_resize_factor*j);
					temp_v = (vec_val[0] + vec_val[1] + vec_val[2])/3;
					temp_v /= 255;
				}
				resized.at<float>(i,j) = alpha*view_host_.at<float>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j) + (1-alpha)*temp_v;
			}
		}
		else
		{
			resized = view_host_;
		}

		std::string window_name = "view"+ to_string(view_id);
		cv::imshow(window_name, resized);
//		cv::moveWindow(window_name,win_xpos,win_ypos);

		if (MY_CONFIG::seq_direct_saving)
		{
			// Sequence Saving
			view_device_.download(view_host_.ptr<void>(), view_host_.step);
			view_host_ = view_host_* MY_CONFIG::multiplying_factor_for_save_depth_image;
			cv::Mat save_mat(view_host_.rows, view_host_.cols, CV_16U);
			for (int i = 0 ; i < view_host_.rows ; i++)
			for (int j = 0 ; j < view_host_.cols ; j++)
			{
				ushort d = (ushort)view_host_.at<float>(i,j);
				save_mat.at<ushort>(i,j) = (ushort)view_host_.at<float>(i,j);
			}

			cv::imwrite("result/view"+to_string(view_id) +"_" + to_string(seq_count) +".png", save_mat);
			std::cout << "seq save" << std::endl;
		}
    }
    void save_generated_depth(KinFu& kinfu, Affine3f cur_view, int view_id, const int mode)
    {
    	kinfu.renderImage(view_device_, cur_view, mode);
		view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_* MY_CONFIG::multiplying_factor_for_save_depth_image;

		cv::Mat save_mat(view_host_.rows, view_host_.cols, CV_16U);
		for (int i = 0 ; i < view_host_.rows ; i++)
		for (int j = 0 ; j < view_host_.cols ; j++)
		{
			save_mat.at<ushort>(i,j) = (ushort)view_host_.at<float>(i,j);
		}
		cv::imwrite("view"+to_string(view_id)+".png", save_mat);
    }

    Affine3f CamRotation(float _thetax=0, float _thetay=0, float _thetaz=0)
    {
    	Affine3f X;
		X.matrix(0,0) = 1; X.matrix(0,1) = 0; 		     X.matrix(0,2) = 0;
		X.matrix(1,0) = 0; X.matrix(1,1) = cos(_thetax); X.matrix(1,2) = -sin(_thetax);
		X.matrix(2,0) = 0; X.matrix(2,1) = sin(_thetax); X.matrix(2,2) = cos(_thetax);

		Affine3f Y;
		Y.matrix(0,0) = cos(_thetay);  Y.matrix(0,1) = 0; Y.matrix(0,2) = sin(_thetay);
		Y.matrix(1,0) = 0; 		       Y.matrix(1,1) = 1; Y.matrix(1,2) = 0;
		Y.matrix(2,0) = -sin(_thetay); Y.matrix(2,1) = 0; Y.matrix(2,2) = cos(_thetay);

		Affine3f Z;
		Z.matrix(0,0) = cos(_thetaz);  Z.matrix(0,1) = -sin(_thetaz); Z.matrix(0,2) = 0;
		Z.matrix(1,0) = sin(_thetaz);  Z.matrix(1,1) = cos(_thetaz);  Z.matrix(1,2) = 0;
		Z.matrix(2,0) = 0; 			   Z.matrix(2,1) = 0; 		      Z.matrix(2,2) = 1;

		return Z*Y*X;
    }

    void show_raycasted(KinFu& kinfu, int seq_count, std::string seq_id)
    {
        const int mode = 1;

        // initialize
        Affine3f origin_view = viz.getViewerPose();
        Affine3f cur_view = origin_view;

        int x_sign = -1;
        int y_sign = -1;
        int z_sign = -1;

        double rel_diff[4][3];
        rel_diff[0][0] =  320.071886/1000 * x_sign;
        rel_diff[0][1] = -395.942208/1000 * y_sign;
        rel_diff[0][2] =  -47.552504/1000 * z_sign;

        rel_diff[1][0] =  262.323648/1000 * x_sign;
        rel_diff[1][1] = -393.273527/1000 * y_sign;
        rel_diff[1][2] =   -43.39385/1000 * z_sign;

        rel_diff[2][0] =  191.069625/1000 * x_sign;
		rel_diff[2][1] = -384.284323/1000 * y_sign;
		rel_diff[2][2] =  -41.512631/1000 * z_sign;

		rel_diff[3][0] =  155.531169/1000 * x_sign;
		rel_diff[3][1] = -380.292326/1000 * y_sign;
		rel_diff[3][2] =  -39.278644/1000 * z_sign;

	    // display depth
		{
//			cur_view = CamRotation(-0.01,0,0);
   			cur_view.matrix(0,3) = origin_view.matrix(0,3) + rel_diff[0][0];
			cur_view.matrix(1,3) = origin_view.matrix(1,3) + rel_diff[0][1];
			cur_view.matrix(2,3) = origin_view.matrix(2,3) + rel_diff[0][2];
			display_generated_depth(kinfu, cur_view, 1, mode, 100,100, 1, seq_count, seq_id);

			cur_view.matrix(0,3) = origin_view.matrix(0,3) + rel_diff[3][0];
			cur_view.matrix(1,3) = origin_view.matrix(1,3) + rel_diff[3][1];
			cur_view.matrix(2,3) = origin_view.matrix(2,3) + rel_diff[3][2];
			display_generated_depth(kinfu, cur_view, 4, mode, 1150,100, 4, seq_count, seq_id);


			cur_view = CamRotation(-0.02,0,0);
			cur_view.matrix(0,3) = origin_view.matrix(0,3) + rel_diff[1][0];
			cur_view.matrix(1,3) = origin_view.matrix(1,3) + rel_diff[1][1];
			cur_view.matrix(2,3) = origin_view.matrix(2,3) + rel_diff[1][2];
			display_generated_depth(kinfu, cur_view, 2, mode, 450,100, 2, seq_count, seq_id);

			cur_view.matrix(0,3) = origin_view.matrix(0,3) + rel_diff[2][0];
			cur_view.matrix(1,3) = origin_view.matrix(1,3) + rel_diff[2][1];
			cur_view.matrix(2,3) = origin_view.matrix(2,3) + rel_diff[2][2];
			display_generated_depth(kinfu, cur_view, 3, mode, 800,100, 3, seq_count, seq_id);
		}
    }

    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
//        viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }

    void config_by_seq(KinFu& kinfu, std::string seq_id)
    {
    	if(!MY_CONFIG::use_all_frames)
    	{
    		if (seq_id == "20161204_231541")
			{
				start_frame = 30;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.8;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
    		else if (seq_id == "20161204_232513")
			{
				start_frame = 60;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 1.0;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
			else if (seq_id == "20161204_232836")
			{
				start_frame = 30;
				end_frame = 150;
				kinfu.params_.icp_dist_thres = 0.3;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else if (seq_id == "20161204_233533")
			{
				start_frame = 33;
				end_frame = 80;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else if (seq_id == "20170113_142649")
			{
				start_frame = 0;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else if (seq_id == "20170113_143623")
			{
				start_frame = 0;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else if (seq_id == "20170114_001102")
			{
				start_frame = 0;
				end_frame = 140;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else
			{
				start_frame = 0;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
    	}
    	else if(MY_CONFIG::use_all_frames)
		{
    		if (seq_id == "20161204_231541")
			{
				start_frame = 0;
				end_frame = 174;
				kinfu.params_.icp_dist_thres = 0.8;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
    		else if (seq_id == "20161204_232513")
			{
				start_frame = 0;
				end_frame = 197;
				kinfu.params_.icp_dist_thres = 1.0;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
			else if (seq_id == "20161204_232836")
			{
				start_frame = 0;
				end_frame = 195;
				kinfu.params_.icp_dist_thres = 0.3;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else if (seq_id == "20161204_233533")
			{
				start_frame = 0;
				end_frame = 109;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
			else if (seq_id == "20170113_142649")
			{
				start_frame = 0;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
			else if (seq_id == "20170113_143623")
			{
				start_frame = 0;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
			else if (seq_id == "20170114_001102")
			{
				start_frame = 0;
				end_frame = 140;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.5f;
			}
			else
			{
				start_frame = 0;
				end_frame = 100;
				kinfu.params_.icp_dist_thres = 0.9;
				kinfu.params_.icp_truncate_depth_dist = 3.9f;
			}
		}
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        config_by_seq(kinfu, MY_CONFIG::seq_id);
        int view_count = 1;
        int seq_count = start_frame;
        int input_source = 1;
        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
        	// input source
        	if (input_source == 0) // kinect direct input
        	{
				bool has_frame = capture_.grab(depth, image);
				if (!has_frame)
					return std::cout << "Can't grab" << std::endl, false;
        	}
        	else { // sequence input
				if (view_count > MY_CONFIG::max_view_num) {
					view_count = 1;
					kinfu.reset();
					seq_count++;
				}
				// circular operation
				if (seq_count > end_frame)
					seq_count = start_frame;

				depth_seq_filename(MY_CONFIG::seq_id, view_count, seq_count);
				depth = cv::imread(depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);

				view_count++;
        	}

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = kinfu(depth_device_);
            }

            if (has_image)
            {
                show_raycasted(kinfu, seq_count, MY_CONFIG::seq_id);
                std::cout << "seq_count : " << seq_count << std::endl;
            }

            show_depth(depth);

            if (!iteractive_mode_)
            	viz.setViewerPose(kinfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);

            switch(key)
            {
				case 't': case 'T' : take_cloud(kinfu); break;
				case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
//				case 'c': case 'C' : depth_gen = true; break;
				case 27: exit_ = true; break;
				case 32: pause_ = !pause_; break;
            }

            //exit_ = exit_ || i > 100;
            viz.spinOnce(3, true);
        }
        return true;
    }

	float theta_Z;
	float theta_Y;
	float theta_X;

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    bool depth_gen = false;

    OpenNISource& capture_;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;

    float blending_alpha=0.8;
    std::string color_file_name;
    std::string depth_file_name;

    int start_frame;
    int end_frame;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = MY_CONFIG::device_num;
    cuda::setDevice (device);
    cuda::printCudaDeviceInfo (device);

    OpenNISource capture;
    capture.open (0);

    KinFuApp app (capture);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& e) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
