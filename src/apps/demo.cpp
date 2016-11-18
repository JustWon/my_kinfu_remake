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
		cv::moveWindow(window_name,win_xpos,win_ypos);
    }
    void diplay_generated_depth(KinFu& kinfu, Affine3f cur_view, int view_id, const int mode, int win_xpos, int win_ypos, int view_count, int seq_count)
    {
    	kinfu.renderImage(view_device_, cur_view, mode);
    	view_host_.create(view_device_.rows(), view_device_.cols(), CV_32FC1);
    	view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_/ MY_CONFIG::vh_dividing_factor;
		cv::Mat resized(view_host_.rows/MY_CONFIG::vh_resize_factor, view_host_.cols/MY_CONFIG::vh_resize_factor,CV_32FC1);

		// temp code
		std::string file_name = "/home/dongwonshin/Desktop/20161118 test seq/com1/20161118_004252/sequence/color"+to_string(view_count)+"_"+ to_string(seq_count)+".bmp";
		cv::Mat here_img = cv::imread(file_name);
		cv::flip(here_img, here_img, 1);

		float alpha = 0.5;
		for (int i = 0 ; i < view_host_.rows/MY_CONFIG::vh_resize_factor ; i++)
		for (int j = 0 ; j < view_host_.cols/MY_CONFIG::vh_resize_factor ; j++)
		{
			float temp_v=0;
			if (!here_img.empty())
			{
				cv::Vec3b vec_val = here_img.at<cv::Vec3b>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j);
				temp_v = (vec_val[0] + vec_val[1] + vec_val[2])/3;
				temp_v /= 255;
			}
			resized.at<float>(i,j) = alpha*view_host_.at<float>(MY_CONFIG::vh_resize_factor*i,MY_CONFIG::vh_resize_factor*j) + alpha*temp_v;
		}
		std::string window_name = "view"+ to_string(view_id);
		cv::imshow(window_name, resized);
		cv::moveWindow(window_name,win_xpos,win_ypos);

    }
    void save_generated_depth(KinFu& kinfu, Affine3f cur_view, int view_id, const int mode)
    {
    	kinfu.renderImage(view_device_, cur_view, mode);
		view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_* MY_CONFIG::multiplying_factor_for_save_depth_image;
		cv::imwrite("view"+to_string(view_id)+".bmp", view_host_);

		cv::FileStorage storage("view"+to_string(view_id)+".yml", cv::FileStorage::WRITE);
		storage << "img" << view_host_;
		storage.release();
    }

    void show_raycasted(KinFu& kinfu, int seq_count)
    {
        const int mode = 1;

        // initialize
        Affine3f cur_view = viz.getViewerPose();

        // display depth
        {
			diplay_generated_depth(kinfu, cur_view, 0, mode, 100,100, 0, seq_count);
			cur_view.matrix(0,3) += -60.118894/1000;
			cur_view.matrix(1,3) += -1.061838/1000;
			cur_view.matrix(2,3) += 3.539284/1000;
			diplay_generated_depth(kinfu, cur_view, 1, mode, 450,100, 1, seq_count);
			cur_view.matrix(0,3) += -67.30809/1000;
			cur_view.matrix(1,3) += -7.107205/1000;
			cur_view.matrix(2,3) += -4.976931/1000;
			diplay_generated_depth(kinfu, cur_view, 2, mode, 800,100, 2, seq_count);
			cur_view.matrix(0,3) += -38.334795/1000;
			cur_view.matrix(1,3) += -3.936116/1000;
			cur_view.matrix(2,3) += -6.865113/1000;
			diplay_generated_depth(kinfu, cur_view, 3, mode, 1150,100, 3, seq_count);
        }

		// save depth
        if (depth_gen) {
        	std::cout << "depth image capture" << std::endl;
        	Affine3f cur_view = viz.getViewerPose();

        	save_generated_depth(kinfu, cur_view, 0, mode);
//			cur_view.matrix(0,3) += 40.074353/1000;
//			cur_view.matrix(1,3) += -1.973618/1000;
//			cur_view.matrix(2,3) += -5.539809/1000;
//			save_generated_depth(kinfu, cur_view, 1, mode);
//			cur_view.matrix(0,3) += 70.204512/1000;
//			cur_view.matrix(1,3) += -6.869942/1000;
//			cur_view.matrix(2,3) += -3.684597/1000;
//			save_generated_depth(kinfu, cur_view, 2, mode);
//			cur_view.matrix(0,3) += 61.193986/1000;
//			cur_view.matrix(1,3) += -2.879078/1000;
//			cur_view.matrix(2,3) += -9.842761/1000;
//			save_generated_depth(kinfu, cur_view, 3, mode);

        	depth_gen = false;
        }
    }

    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        int view_count = 1;
        int seq_count = 0;
        int input_source = 1;
        cv::Mat temp_color_img;
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
				if (seq_count > MY_CONFIG::max_seq_num)
				{
					seq_count = 0;
				}

				//std::string file_name = "/home/dongwonshin/Desktop/20161031 test/com"+ to_string(view_count) +"/20161031_005318 sequence/sequence/kinect_depth"+to_string(seq_count)+".png";
				std::string file_name = "/home/dongwonshin/Desktop/20161118 test seq/com"+ to_string(view_count) +"/20161118_004252/sequence/kinect_depth"+to_string(seq_count)+".png";
				cv::Mat tof_img = cv::imread(file_name, CV_LOAD_IMAGE_ANYDEPTH);

				depth = tof_img/MY_CONFIG::dividing_factor_for_seq_tof_img;

				view_count++;
        	}

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = kinfu(depth_device_);
            }

            show_depth(depth);
            if (has_image)
            {
                show_raycasted(kinfu, seq_count);
                std::cout << "seq_count : " << seq_count << std::endl;
            }

            if (!iteractive_mode_)
            {
            	kfusion::Affine3f temp = kinfu.getCameraPose();
            	{
					temp.matrix(0,0) =  0.999727; temp.matrix(0,1) = -0.012631; temp.matrix(0,2) = 0.019679;
					temp.matrix(1,0) =  0.013289; temp.matrix(1,1) =  0.999344; temp.matrix(1,2) =-0.033677;
					temp.matrix(2,0) = -0.019240; temp.matrix(2,1) =  0.033929; temp.matrix(2,2) = 0.999239;

					temp.matrix(0,3) += 172.663979/1000;
					temp.matrix(1,3) += 409.982024/1000;
					temp.matrix(2,3) += 32.027919/1000;
            	}
                viz.setViewerPose(temp);
            }

            int key = cv::waitKey(pause_ ? 0 : 3);

            switch(key)
            {
				case 't': case 'T' : take_cloud(kinfu); break;
				case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
				case 'c': case 'C' : depth_gen = true; break;
				case 27: exit_ = true; break;
				case 32: pause_ = !pause_; break;
            }

            //exit_ = exit_ || i > 100;
            viz.spinOnce(3, true);
        }
        return true;
    }

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
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
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
