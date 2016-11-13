#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>
#include <stdio.h>
#include <string>

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

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 1;
        int resize_factor = 4;

        Affine3f cur_view = viz.getViewerPose();
        kinfu.renderImage(view_device_, cur_view, mode);
        view_host_.create(view_device_.rows(), view_device_.cols(), CV_32FC1);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        view_host_ = view_host_/3;
        cv::Mat resized(view_host_.rows/resize_factor, view_host_.cols/resize_factor,CV_32FC1);
        for (int i = 0 ; i < view_host_.rows/resize_factor ; i++)
        for (int j = 0 ; j < view_host_.cols/resize_factor ; j++)
        	resized.at<float>(i,j) = view_host_.at<float>(resize_factor*i,resize_factor*j);
        cv::imshow("view0", resized);

		cur_view.matrix(0,3) += 40.074353/1000;
		cur_view.matrix(1,3) += -1.973618/1000;
		cur_view.matrix(2,3) += -5.539809/1000;
		kinfu.renderImage(view_device_, cur_view, mode);
		view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_/3;
		for (int i = 0 ; i < view_host_.rows/resize_factor ; i++)
		for (int j = 0 ; j < view_host_.cols/resize_factor ; j++)
			resized.at<float>(i,j) = view_host_.at<float>(resize_factor*i,resize_factor*j);
		cv::imshow("view1",resized);

		cur_view.matrix(0,3) += 70.204512/1000;
		cur_view.matrix(1,3) += -6.869942/1000;
		cur_view.matrix(2,3) += -3.684597/1000;
		kinfu.renderImage(view_device_, cur_view, mode);
		view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_/3;
		for (int i = 0 ; i < view_host_.rows/resize_factor ; i++)
		for (int j = 0 ; j < view_host_.cols/resize_factor ; j++)
			resized.at<float>(i,j) = view_host_.at<float>(resize_factor*i,resize_factor*j);
		cv::imshow("view2",resized);

		cur_view.matrix(0,3) += 61.193986/1000;
		cur_view.matrix(1,3) += -2.879078/1000;
		cur_view.matrix(2,3) += -9.842761/1000;
		kinfu.renderImage(view_device_, cur_view, mode);
		view_device_.download(view_host_.ptr<void>(), view_host_.step);
		view_host_ = view_host_/3;
		for (int i = 0 ; i < view_host_.rows/resize_factor ; i++)
		for (int j = 0 ; j < view_host_.cols/resize_factor ; j++)
			resized.at<float>(i,j) = view_host_.at<float>(resize_factor*i,resize_factor*j);
		cv::imshow("view3",resized);



        if (depth_gen) {
        	std::cout << "depth image capture" << std::endl;
        	Affine3f cur_view = viz.getViewerPose();

			kinfu.renderImage(view_device_, cur_view, mode);
			view_device_.download(view_host_.ptr<void>(), view_host_.step);
			cv::imwrite("view0.bmp", view_host_);

			cv::FileStorage storage("view0.yml", cv::FileStorage::WRITE);
			storage << "img" << view_host_;
			storage.release();

//			cur_view.matrix(0,3) += 40.074353/1000;
//			cur_view.matrix(1,3) += -1.973618/1000;
//			cur_view.matrix(2,3) += -5.539809/1000;
//			kinfu.renderImage(view_device_, cur_view, mode);
//			view_device_.download(view_host_.ptr<void>(), view_host_.step);
//			cv::imwrite("view1.bmp",view_host_);
//
//			cur_view.matrix(0,3) += 70.204512/1000;
//			cur_view.matrix(1,3) += -6.869942/1000;
//			cur_view.matrix(2,3) += -3.684597/1000;
//			kinfu.renderImage(view_device_, cur_view, mode);
//			view_device_.download(view_host_.ptr<void>(), view_host_.step);
//			cv::imwrite("view2.bmp",view_host_);
//
//			cur_view.matrix(0,3) += 61.193986/1000;
//			cur_view.matrix(1,3) += -2.879078/1000;
//			cur_view.matrix(2,3) += -9.842761/1000;
//			kinfu.renderImage(view_device_, cur_view, mode);
//			view_device_.download(view_host_.ptr<void>(), view_host_.step);
//			cv::imwrite("view3.bmp",view_host_);

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
        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
            bool has_frame = capture_.grab(depth, image);
            if (!has_frame)
                return std::cout << "Can't grab" << std::endl, false;

            // kinect v2 tof image experiment
			if (view_count > 2) {
				view_count = 1;
				kinfu.reset();
			}
			if (seq_count > 489)
			{
				seq_count = 0;
			}
            char file_name[1024];
            sprintf(file_name,"/home/dongwonshin/Desktop/20161031 test/com%d/20161031_005318 sequence/sequence/kinect_depth%d.png", view_count, seq_count);
			view_count++;
			seq_count++;
			cv::Mat tof_img = cv::imread(file_name, CV_LOAD_IMAGE_ANYDEPTH);

			for (int y = 0 ; y < tof_img.rows ; y++)
			for (int x = 0 ; x < tof_img.cols ; x++)
			{
				tof_img.at<ushort>(y,x) /= 3;
			}
			depth = tof_img;

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = kinfu(depth_device_);
            }

            if (has_image)
                show_raycasted(kinfu);

            show_depth(depth);
            //cv::imshow("Image", image);

            if (!iteractive_mode_)
            {
            	kfusion::Affine3f temp = kinfu.getCameraPose();
            	float rx = -272.330416;
            	float ry = 454.777229;
            	float rz = -219.745544;
            	temp.matrix(0,3) += rx/10000;
            	temp.matrix(1,3) += ry/10000;
            	temp.matrix(2,3) += rz/1000;
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
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    OpenNISource capture;
    capture.open (0);

    KinFuApp app (capture);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& e) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
