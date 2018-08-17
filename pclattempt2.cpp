#include <pcl/io/pcd_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/feature.h>
#include <pcl/features/boundary.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/pcl_base.h>
#include <pcl/correspondence.h>
#include <boost/unordered_map.hpp>
#include <pcl/registration/correspondence_types.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
/*List of references and citations
1. Radu Bogdan Rusu,Steve Cousins "3D is here:Point Cloud Library",IEEE International Conference on Robotics and Automation 2011 pp 1-4(Functions provided in PCL are used to read point clouds,view point clouds,estimate boundary points,kdtree,FlannMatching
ICP and finally SAMPLE CONSENSUS MODEL ALGORITHM :RANSAC
2. Marc Alexa and Adams Anderson "On Normals and Projection Operators for Surfaces Defined by Point Sets
3.N J Mitra and A Nguyen"Estimating Surface Normals in Noisy Point Cloud Data" in SCG 2003 pp322-328
4. S Holzer,Radu Bogdan Rusu,M Dixon,S Gedikli and N Navab "Adaptive Neigbhourhood Selection For Real Time Surface Normal Estimation from Organized Point Cloud Data using Integral Images International Conference on Intelligent Robots and Systems 2012
5.Elsberg J,B Dorrit,N Andreas "One Billion Points in the cloud  An Octree for efficient processing of 3D Laser Scans" Journal Of Photogrammetry and Remote Sensing No 76  Feb 2013 pp76-88
6. Fengjun Hu,Yanwei Zhao,Wanliang Wang and Xianping Huang "Discrete Point Cloud Filtering and Searching Based on VGSO Algorithm" Proceedings of 27th European Conference on Modelling and Simulation 2013
7. Radu Bogdan Rusu,Zoltan Csaba Marton,Nico Blodow and Michael Beetz "Persistent Point Feature Histograms for 3D Point Clouds" Proceedings of 10th International Conference on Intelligent Autonomous Systems 2008
8. Radu Bogdan Rusu,Nico Blodow,Michael Beetz "Fast Point Feature Histograms for 3D Registration" IEEE International Conference on Robotics and Automation 2009
9.M Muja and D.G.Lowe "Fast Approximate Nearest Neighbours with automatic Algorithm Configuration " Proceedings of International Conference on Computer Vision Theory and Applications 2009 pp 331-340
10. Liu Ran,Wan Wanggen,Zhou Yivuan,Lu Libling,Zhang Ximin "Normal Estimation Algorithm for point cloud using KD Tree" IET International Conference on Smart and Sustainable City 2013 (Used in lines 
11. M.A Fischler and R.C Bolles "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography" in Communications of the ACM Vol 24 1981 pp 381-395(Used in lines 171-305)
12. Besl Paul J,N.D. McKay "A Method for Registration of 3D Shapes ":IEEE Transacations on Pattern Analysis and Machine Intelligence 1992 pp239-256
13.Gael Guennebaud et al "Eigen A C++ Linear Algebra Library http://eigen.tuxfamily.org"
14.Beman Dawes and David Abrahams "http://www.boost.org/"
15. Will Schroeder,Ken Martin and Bill Lorensen "VTK: Visualization Toolkit "https://www.vtk.org/Wiki/VTK"
*/
struct CloudStyle
{
	double r;
	double g;
	double b;
	double size;

	CloudStyle(double r,
		double g,
		double b,
		double size) :
		r(r),
		g(g),
		b(b),
		size(size)
	{
	}
};

CloudStyle style_white(255.0, 255.0, 255.0, 4.0);
CloudStyle style_red(255.0, 0.0, 0.0, 3.0);
CloudStyle style_green(0.0, 255.0, 0.0, 5.0);
CloudStyle style_cyan(93.0, 200.0, 217.0, 4.0);
CloudStyle style_violet(255.0, 0.0, 255.0, 8.0);
double
computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	double resolution = 0.0;
	int numberOfPoints = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (! pcl_isfinite((*cloud)[i].x))
			continue;

		// Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (nres == 2)
		{
			resolution += sqrt(squaredDistances[1]);
			++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;

	return resolution;
}
namespace pcl
{
	namespace registration
	{
		template <typename PointT>
		class CorrespondenceRejectorSampleConsensus : public CorrespondenceRejector
		{
			typedef pcl::PointCloud<PointT> PointCloud;
			typedef typename PointCloud::Ptr PointCloudPtr;
			typedef typename PointCloud::ConstPtr PointCloudConstPtr;

		public:
			using CorrespondenceRejector::input_correspondences_;
			using CorrespondenceRejector::rejection_name_;
			using CorrespondenceRejector::getClassName;

			typedef boost::shared_ptr<CorrespondenceRejectorSampleConsensus> Ptr;
			typedef boost::shared_ptr<const CorrespondenceRejectorSampleConsensus> ConstPtr;

			/** \brief Empty constructor. Sets the inlier threshold to 5cm (0.05m),
			* and the maximum number of iterations to 1000.
			//for milk set it to 0.075
			//for clorox it was 0.015
			//for wine 0.015
			for mug trying 0.025
			*/
			CorrespondenceRejectorSampleConsensus()
				: inlier_threshold_(0.35)
				, max_iterations_(1000) // std::numeric_limits<int>::max ()
				, input_()
				, input_transformed_()
				, target_()
				, best_transformation_()
				, refine_(false)
				, save_inliers_(false)
			{
				rejection_name_ = "CorrespondenceRejectorSampleConsensus";
			}

			/** \brief Empty destructor. */
			virtual ~CorrespondenceRejectorSampleConsensus() {}
			inline void
				getRemainingCorrespondences(const pcl::Correspondences& original_correspondences,
					pcl::Correspondences& remaining_correspondences) {
				if (!input_)
				{
					PCL_ERROR("[pcl::registration::%s::getRemainingCorrespondences] No input cloud dataset was given!\n", getClassName().c_str());
					return;
				}

				if (!target_)
				{
					PCL_ERROR("[pcl::registration::%s::getRemainingCorrespondences] No input target dataset was given!\n", getClassName().c_str());
					return;
				}

				if (save_inliers_)
					inlier_indices_.clear();

				int nr_correspondences = static_cast<int> (original_correspondences.size());
				std::vector<int> source_indices(nr_correspondences);
				std::vector<int> target_indices(nr_correspondences);

				// Copy the query-match indices
				for (size_t i = 0; i < original_correspondences.size(); ++i)
				{
					source_indices[i] = original_correspondences[i].index_query;
					target_indices[i] = original_correspondences[i].index_match;
				}

				// from pcl/registration/icp.hpp:
				std::vector<int> source_indices_good;
				std::vector<int> target_indices_good;
				{
					// From the set of correspondences found, attempt to remove outliers
					// Create the registration model
					//Sample Consensus Model Registration and RANSAC mentioned in citations. 
					typedef typename pcl::SampleConsensusModelRegistration<PointT>::Ptr SampleConsensusModelRegistrationPtr;
					SampleConsensusModelRegistrationPtr model;
					model.reset(new pcl::SampleConsensusModelRegistration<PointT>(input_, source_indices));
					// Pass the target_indices
					model->setInputTarget(target_, target_indices);
					// Create a RANSAC model
					pcl::RandomSampleConsensus<PointT> sac(model, inlier_threshold_);
					sac.setMaxIterations(max_iterations_);

					// Compute the set of inliers
					if (!sac.computeModel())
					{
						remaining_correspondences = original_correspondences;
						best_transformation_.setIdentity();
						return;
					}
					else
					{
						if (refine_ && !sac.refineModel())
						{

							return;
						}

						std::vector<int> inliers;
						sac.getInliers(inliers);

						if (inliers.size() < 3)
						{
							remaining_correspondences = original_correspondences;
							best_transformation_.setIdentity();
							return;
						}
						boost::unordered_map<int, int> index_to_correspondence;
						for (int i = 0; i < nr_correspondences; ++i)
							index_to_correspondence[original_correspondences[i].index_query] = i;

						remaining_correspondences.resize(inliers.size());
						for (size_t i = 0; i < inliers.size(); ++i)
							remaining_correspondences[i] = original_correspondences[index_to_correspondence[inliers[i]]];

						if (save_inliers_)
						{
							inlier_indices_.reserve(inliers.size());
							for (size_t i = 0; i < inliers.size(); ++i)
								inlier_indices_.push_back(index_to_correspondence[inliers[i]]);
						}

						// get best transformation
						Eigen::VectorXf model_coefficients;
						sac.getModelCoefficients(model_coefficients);
						best_transformation_.row(0) = model_coefficients.segment<4>(0);
						best_transformation_.row(1) = model_coefficients.segment<4>(4);
						best_transformation_.row(2) = model_coefficients.segment<4>(8);
						best_transformation_.row(3) = model_coefficients.segment<4>(12);
					}
				}
			}
			virtual inline void
				setInputSource(const PointCloudConstPtr &cloud)
			{
				input_ = cloud;
			}
			inline PointCloudConstPtr const
				getInputSource() { return (input_); }
			virtual inline void
				setInputTarget(const PointCloudConstPtr &cloud) { target_ = cloud; }
			inline PointCloudConstPtr const
				getInputTarget() { return (target_); }
			bool
				requiresSourcePoints() const
			{
				return (true);
			}
			void
				setSourcePoints(pcl::PCLPointCloud2::ConstPtr cloud2)
			{
				PointCloudPtr cloud(new PointCloud);
				fromPCLPointCloud2(*cloud2, *cloud);
				setInputSource(cloud);
			}
			bool
				requiresTargetPoints() const
			{
				return (true);
			}
			void
				setTargetPoints(pcl::PCLPointCloud2::ConstPtr cloud2)
			{
				PointCloudPtr cloud(new PointCloud);
				fromPCLPointCloud2(*cloud2, *cloud);
				setInputTarget(cloud);
			}
			inline void
				setInlierThreshold(double threshold) { inlier_threshold_ = threshold; };
			inline double
				getInlierThreshold() { return inlier_threshold_; };
			inline void
				setMaximumIterations(int max_iterations) { max_iterations_ = std::max(max_iterations, 0); }
			inline int
				getMaximumIterations() { return (max_iterations_); }
			inline Eigen::Matrix4f
				getBestTransformation() { return best_transformation_; };
			inline void
				setRefineModel(const bool refine)
			{
				refine_ = refine;
			}
			inline bool
				getRefineModel() const
			{
				return (refine_);
			}
			inline void
				getInliersIndices(std::vector<int> &inlier_indices) { inlier_indices = inlier_indices_; }
			inline void
				setSaveInliers(bool s) { save_inliers_ = s; }
			inline bool
				getSaveInliers() { return save_inliers_; }
		protected:

			inline void
				applyRejection(pcl::Correspondences &correspondences)
			{
				getRemainingCorrespondences(*input_correspondences_, correspondences);
			}
			double inlier_threshold_;
			int max_iterations_;
			PointCloudConstPtr input_;
			PointCloudPtr input_transformed_;
			PointCloudConstPtr target_;
			Eigen::Matrix4f best_transformation_;
			bool refine_;
			std::vector<int> inlier_indices_;
			bool save_inliers_;
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		};

	}
	template <typename PointModelT, typename PointSceneT>
	class CorresGroup : public PCLBase<PointModelT>
	{
	public:
		typedef pcl::PointCloud<PointSceneT> SceneCloud;
		typedef typename SceneCloud::Ptr SceneCloudPtr;
		typedef typename SceneCloud::ConstPtr SceneCloudConstPtr;


		CorresGroup() : scene_(), model_scene_corrs_() {}
		virtual ~CorresGroup()
		{
			scene_.reset();
			model_scene_corrs_.reset();
		}
		virtual inline void
			setSceneCloud(const SceneCloudConstPtr &scene)
		{
			scene_ = scene;
		}

		/** \brief Getter for the scene dataset.
		*
		* \return the const boost shared pointer to a PointCloud message.
		*/
		inline SceneCloudConstPtr
			getSceneCloud() const
		{
			return (scene_);
		}
		virtual inline void
			setModelSceneCorrespondences(const CorrespondencesConstPtr &corrs)
		{
			model_scene_corrs_ = corrs;
		}
		inline CorrespondencesConstPtr
			getModelSceneCorrespondences() const
		{
			return (model_scene_corrs_);
		}
		inline std::vector<double>
			getCharacteristicScales() const
		{
			return (corr_group_scale_);
		}
		void
			cluster(std::vector<Correspondences> &clustered_corrs) {
			clustered_corrs.clear();
			corr_group_scale_.clear();
			if (!initCompute())
			{
				return;
			}
			clusterCorrespondences(clustered_corrs);
			deinitCompute();
		}

	protected:

		SceneCloudConstPtr scene_;
		using PCLBase<PointModelT>::input_;
		CorrespondencesConstPtr model_scene_corrs_;
		virtual void
			clusterCorrespondences(std::vector<Correspondences> &clustered_corrs) = 0;
		inline bool
			initCompute()
		{
			if (!PCLBase<PointModelT>::initCompute())
			{
				return (false);
			}

			if (!scene_)
			{
				PCL_ERROR("[initCompute] Scene not set.\n");
				return (false);
			}

			if (!input_)
			{
				PCL_ERROR("[initCompute] Input not set.\n");
				return (false);
			}

			if (!model_scene_corrs_)
			{
				PCL_ERROR("[initCompute] Model-Scene Correspondences not set.\n");
				return (false);
			}

			return (true);
		}
		inline bool
			deinitCompute()
		{
			return (true);
		}

	};



	//The base of this class serves as a foundation for the actual correspondence grouping that we will
	// be performing.
	template<typename PointModelT, typename PointSceneT>
	class GeometricConsistencyGrouping : public CorresGroup<PointModelT, PointSceneT>
	{
	public:
		typedef pcl::PointCloud<PointModelT> PointCloud;
		typedef typename PointCloud::Ptr PointCloudPtr;
		typedef typename PointCloud::ConstPtr PointCloudConstPtr;

		typedef typename pcl::CorresGroup<PointModelT, PointSceneT>::SceneCloudConstPtr SceneCloudConstPtr;
		GeometricConsistencyGrouping()
			: gcthresh(3)
			, gcsize(1.0)
			, found_transformations_()
		{}
		inline void
			setGCThreshold(int threshold)
		{
			gcthresh = threshold;
		}
		inline int
			getGCThreshold() const
		{
			return (gcthresh);
		}
		inline void
			setGCSize(double gc_size)
		{
			gcsize = gc_size;
		}
		inline double
			getGCSize() const
		{
			return (gcsize);
		}
		bool
			recognize(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations) {
			std::vector<pcl::Correspondences> model_instances;
			return (this->recognize(transformations, model_instances));
		}
		bool
			recognize(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations, std::vector<pcl::Correspondences> &clustered_corrs) {
			transformations.clear();
			if (!this->initCompute())
			{
				return(false);
			}
			clusterCorrespondences(clustered_corrs);
			transformations = found_transformations_;
			this->deinitCompute();
			return (true);

		}

	protected:
		using CorresGroup<PointModelT, PointSceneT>::input_;
		using CorresGroup<PointModelT, PointSceneT>::scene_;
		using CorresGroup<PointModelT, PointSceneT>::model_scene_corrs_;
		int gcthresh;

		double gcsize;
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > found_transformations_;
		void
			clusterCorrespondences(std::vector<Correspondences> &model_instances);

	};
	bool
		gcCorrespSorter(pcl::Correspondence i, pcl::Correspondence j)
	{
		return (i.distance < j.distance);
	}
	template<typename PointModelT, typename PointSceneT> void
		pcl::GeometricConsistencyGrouping<PointModelT, PointSceneT>::clusterCorrespondences(std::vector<Correspondences> &model_instances)
	{
		model_instances.clear();
		found_transformations_.clear();

		if (!model_scene_corrs_)
		{
			PCL_ERROR(
				"[pcl::GeometricConsistencyGrouping::clusterCorrespondences()] Error! Correspondences not set, please set them before calling again this function.\n");
			return;
		}

		CorrespondencesPtr sorted_corrs(new Correspondences(*model_scene_corrs_));

		std::sort(sorted_corrs->begin(), sorted_corrs->end(), gcCorrespSorter);

		model_scene_corrs_ = sorted_corrs;

		std::vector<int> consensus_set;
		std::vector<bool> taken_corresps(model_scene_corrs_->size(), false);

		Eigen::Vector3f dist_ref, dist_trg;

		//temp copy of scene cloud with the type cast to ModelT in order to use Ransac
		PointCloudPtr temp_scene_cloud_ptr(new PointCloud());
		pcl::copyPointCloud<PointSceneT, PointModelT>(*scene_, *temp_scene_cloud_ptr);
		//Lines 494-542 perform clustering
		//where in for a given consensus set of size equivalent to GCSize which we will
		// be providing as input,the algo checks if a correspondence can fit inside the cluster.
		//After this is done,we perform RANSAC on the entire set of clustered correspondence
		//To reject any outliers so as to make sure the alignment between the model instances 
		//found from this step and the actual model is good.
		pcl::registration::CorrespondenceRejectorSampleConsensus<PointModelT> corr_rejector;
		corr_rejector.setMaximumIterations(10000);
		corr_rejector.setInlierThreshold(gcsize);
		corr_rejector.setInputSource(input_);
		corr_rejector.setInputTarget(temp_scene_cloud_ptr);

		for (size_t i = 0; i < model_scene_corrs_->size(); ++i)
		{
			if (taken_corresps[i])
				continue;

			consensus_set.clear();
			consensus_set.push_back(static_cast<int> (i));

			for (size_t j = 0; j < model_scene_corrs_->size(); ++j)
			{
				if (j != i && !taken_corresps[j])
				{
					//Let's check if j fits into the current consensus set
					bool is_a_good_candidate = true;
					for (size_t k = 0; k < consensus_set.size(); ++k)
					{
						int scene_index_k = model_scene_corrs_->at(consensus_set[k]).index_match;
						int model_index_k = model_scene_corrs_->at(consensus_set[k]).index_query;
						int scene_index_j = model_scene_corrs_->at(j).index_match;
						int model_index_j = model_scene_corrs_->at(j).index_query;

						const Eigen::Vector3f& scene_point_k = scene_->at(scene_index_k).getVector3fMap();
						const Eigen::Vector3f& model_point_k = input_->at(model_index_k).getVector3fMap();
						const Eigen::Vector3f& scene_point_j = scene_->at(scene_index_j).getVector3fMap();
						const Eigen::Vector3f& model_point_j = input_->at(model_index_j).getVector3fMap();

						dist_ref = scene_point_k - scene_point_j;
						dist_trg = model_point_k - model_point_j;

						double distance = fabs(dist_ref.norm() - dist_trg.norm());

						if (distance > gcsize)
						{
							is_a_good_candidate = false;
							break;
						}
					}

					if (is_a_good_candidate)
						consensus_set.push_back(static_cast<int> (j));
				}
			}

			if (static_cast<int> (consensus_set.size()) > gcthresh)
			{
				Correspondences temp_corrs, filtered_corrs;
				for (size_t j = 0; j < consensus_set.size(); j++)
				{
					temp_corrs.push_back(model_scene_corrs_->at(consensus_set[j]));
					taken_corresps[consensus_set[j]] = true;
				}
				corr_rejector.getRemainingCorrespondences(temp_corrs, filtered_corrs);
				//save transformations for recognize
				found_transformations_.push_back(corr_rejector.getBestTransformation());

				model_instances.push_back(filtered_corrs);
			}
		}
	}




	template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal>
	class ISSKeypoint3D1 : public Keypoint<PointInT, PointOutT>
	{
	public:
		typedef boost::shared_ptr<ISSKeypoint3D1<PointInT, PointOutT, NormalT> > Ptr;
		typedef boost::shared_ptr<const ISSKeypoint3D1<PointInT, PointOutT, NormalT> > ConstPtr;

		typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
		typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;

		typedef typename pcl::PointCloud<NormalT> PointCloudN;
		typedef typename PointCloudN::Ptr PointCloudNPtr;
		typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

		typedef typename pcl::octree::OctreePointCloudSearch<PointInT> OctreeSearchIn;
		typedef typename OctreeSearchIn::Ptr OctreeSearchInPtr;

		using Keypoint<PointInT, PointOutT>::name_;
		using Keypoint<PointInT, PointOutT>::input_;
		using Keypoint<PointInT, PointOutT>::surface_;
		using Keypoint<PointInT, PointOutT>::tree_;
		using Keypoint<PointInT, PointOutT>::search_radius_;
		using Keypoint<PointInT, PointOutT>::search_parameter_;
		using Keypoint<PointInT, PointOutT>::keypoints_indices_;
		ISSKeypoint3D1(double salient_radius = 0.0001)
			: salient_radius_(salient_radius)
			, non_max_radius_(0.0)
			, normal_radius_(0.0)
			, border_radius_(0.0)
			, gamma_21_(0.975)
			, gamma_32_(0.975)
			, third_eigen_value_(0)
			, edge_points_(0)
			, min_neighbors_(5)
			, normals_(new pcl::PointCloud<NormalT>)
			, angle_threshold_(static_cast<float> (M_PI) / 2.0f)
			, threads_(0)
		{
			name_ = "ISSKeypoint3D1";
			search_radius_ = salient_radius_;
		}

		/** \brief Destructor. */
		~ISSKeypoint3D1()
		{
			delete[] third_eigen_value_;
			delete[] edge_points_;
		}


		void setSalientRadius(double salient_radius) {
			salient_radius_ = salient_radius;
		}

		void setNonMaxRadius(double non_max_radius) {
			non_max_radius_ = non_max_radius;
		}

		void setNormalRadius(double normal_radius)
		{
			normal_radius_ = normal_radius;

		}

		void  setBorderRadius(double border_radius) {
			border_radius_ = border_radius;
		}

		void      setThreshold21(double gamma_21) {
			gamma_21_ = gamma_21;
		}

		void      setThreshold32(double gamma_32) {
			gamma_32_ = gamma_32;
		}


		void setMinNeighbors(int min_neighbors) {
			min_neighbors_ = min_neighbors;
		}


		void   setNormals(const PointCloudNConstPtr &normals) {
			normals_ = normals;
		}


		inline void
			setAngleThreshold(float angle)
		{
			angle_threshold_ = angle;
		}

		inline void
			setNumberOfThreads(unsigned int nr_threads = 0) { threads_ = nr_threads; }

	protected:


		bool* getBoundaryPoints(PointCloudIn &input, double border_radius, float angle_threshold) {
			bool* edge_points = new bool[input.size()];

			Eigen::Vector4f u = Eigen::Vector4f::Zero();
			Eigen::Vector4f v = Eigen::Vector4f::Zero();

			pcl::BoundaryEstimation<PointInT, NormalT, pcl::Boundary> boundary_estimator;
			boundary_estimator.setInputCloud(input_);

			int index;
#ifdef _OPENMP
#pragma omp parallel for private(u, v) num_threads(threads_)
#endif
			for (index = 0; index < int(input.points.size()); index++)
			{
				edge_points[index] = false;
				PointInT current_point = input.points[index];

				if (pcl::isFinite(current_point))
				{
					std::vector<int> nn_indices;
					std::vector<float> nn_distances;
					int n_neighbors;

					this->searchForNeighbors(static_cast<int> (index), border_radius, nn_indices, nn_distances);

					n_neighbors = static_cast<int> (nn_indices.size());

					if (n_neighbors >= min_neighbors_)
					{
						boundary_estimator.getCoordinateSystemOnPlane(normals_->points[index], u, v);

						if (boundary_estimator.isBoundaryPoint(input, static_cast<int> (index), nn_indices, u, v, angle_threshold))
							edge_points[index] = true;
					}
				}
			}

			return (edge_points);
		}

		/** \brief Compute the scatter matrix
		*/
		void getScatterMatrix(const int &current_index, Eigen::Matrix3d &cov_m) {
			const PointInT& current_point = (*input_).points[current_index];

			double central_point[3];
			memset(central_point, 0, sizeof(double) * 3);

			central_point[0] = current_point.x;
			central_point[1] = current_point.y;
			central_point[2] = current_point.z;

			cov_m = Eigen::Matrix3d::Zero();

			std::vector<int> nn_indices;
			std::vector<float> nn_distances;
			int n_neighbors;

			this->searchForNeighbors(current_index, salient_radius_, nn_indices, nn_distances);

			n_neighbors = static_cast<int> (nn_indices.size());

			if (n_neighbors < min_neighbors_)
				return;

			double cov[9];
			memset(cov, 0, sizeof(double) * 9);

			for (int n_idx = 0; n_idx < n_neighbors; n_idx++)
			{
				const PointInT& n_point = (*input_).points[nn_indices[n_idx]];

				double neigh_point[3];
				memset(neigh_point, 0, sizeof(double) * 3);

				neigh_point[0] = n_point.x;
				neigh_point[1] = n_point.y;
				neigh_point[2] = n_point.z;

				for (int i = 0; i < 3; i++)
					for (int j = 0; j < 3; j++)
						cov[i * 3 + j] += (neigh_point[i] - central_point[i]) * (neigh_point[j] - central_point[j]);
			}

			cov_m << cov[0], cov[1], cov[2],
				cov[3], cov[4], cov[5],
				cov[6], cov[7], cov[8];
		}

		/** \brief Perform the initial checks before computing the keypoints.
		*  \return true if all the checks are passed, false otherwise
		*/
		bool
			initCompute() {
			if (!Keypoint<PointInT, PointOutT>::initCompute())
			{
				
			}
			if (salient_radius_ <= 0)
			{
				
				return (false);
			}
			if (non_max_radius_ <= 0)
			{
				
				return (false);
			}
			if (gamma_21_ <= 0)
			{
				
				return (false);
			}
			if (gamma_32_ <= 0)
			{
				
				return (false);
			}
			if (min_neighbors_ <= 0)
			{
				
				return (false);
			}

			if (third_eigen_value_)
				delete[] third_eigen_value_;

			third_eigen_value_ = new double[input_->size()];
			memset(third_eigen_value_, 0, sizeof(double) * input_->size());

			if (edge_points_)
				delete[] edge_points_;

			if (border_radius_ > 0.0)
			{
				if (normals_->empty())
				{
					if (normal_radius_ <= 0.)
					{
						
						return (false);
					}

					PointCloudNPtr normal_ptr(new PointCloudN());
					if (input_->height == 1)
					{
						pcl::NormalEstimation<PointInT, NormalT> normal_estimation;
						normal_estimation.setInputCloud(surface_);
						normal_estimation.setRadiusSearch(normal_radius_);
						normal_estimation.compute(*normal_ptr);
					}
					else
					{
						pcl::IntegralImageNormalEstimation<PointInT, NormalT> normal_estimation;
						normal_estimation.setNormalEstimationMethod(pcl::IntegralImageNormalEstimation<PointInT, NormalT>::SIMPLE_3D_GRADIENT);
						normal_estimation.setInputCloud(surface_);
						normal_estimation.setNormalSmoothingSize(5.0);
						normal_estimation.compute(*normal_ptr);
					}
					normals_ = normal_ptr;
				}
				if (normals_->size() != surface_->size())
				{
					
					return (false);
				}
			}
			else if (border_radius_ < 0.0)
			{
				
				return (false);
			}

			return (true);
		}

		/** \brief Detect the keypoints by performing the EVD of the scatter matrix.
		* \param[out] output the resultant cloud of keypoints
		*/
		void
			detectKeypoints(PointCloudOut &output) {
			output.points.clear();

			if (border_radius_ > 0.0)
				edge_points_ = getBoundaryPoints(*(input_->makeShared()), border_radius_, angle_threshold_);

			bool* borders = new bool[input_->size()];

			int index;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
			for (index = 0; index < int(input_->size()); index++)
			{
				borders[index] = false;
				PointInT current_point = input_->points[index];

				if ((border_radius_ > 0.0) && (pcl::isFinite(current_point)))
				{
					std::vector<int> nn_indices;
					std::vector<float> nn_distances;

					this->searchForNeighbors(static_cast<int> (index), border_radius_, nn_indices, nn_distances);

					for (size_t j = 0; j < nn_indices.size(); j++)
					{
						if (edge_points_[nn_indices[j]])
						{
							borders[index] = true;
							break;
						}
					}
				}
			}

#ifdef _OPENMP
			Eigen::Vector3d *omp_mem = new Eigen::Vector3d[threads_];

			for (size_t i = 0; i < threads_; i++)
				omp_mem[i].setZero(3);
#else
			Eigen::Vector3d *omp_mem = new Eigen::Vector3d[1];

			omp_mem[0].setZero(3);
#endif

			double *prg_local_mem = new double[input_->size() * 3];
			double **prg_mem = new double *[input_->size()];

			for (size_t i = 0; i < input_->size(); i++)
				prg_mem[i] = prg_local_mem + 3 * i;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
			for (index = 0; index < static_cast<int> (input_->size()); index++)
			{
#ifdef _OPENMP
				int tid = omp_get_thread_num();
#else
				int tid = 0;
#endif
				PointInT current_point = input_->points[index];

				if ((!borders[index]) && pcl::isFinite(current_point))
				{
					//if the considered point is not a border point and the point is "finite", then compute the scatter matrix
					Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero();
					getScatterMatrix(static_cast<int> (index), cov_m);

					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov_m);

					const double& e1c = solver.eigenvalues()[2];
					const double& e2c = solver.eigenvalues()[1];
					const double& e3c = solver.eigenvalues()[0];

					if (!pcl_isfinite(e1c) || !pcl_isfinite(e2c) || !pcl_isfinite(e3c))
						continue;

					if (e3c < 0)
					{
						PCL_WARN("[pcl::%s::detectKeypoints] :  Skipping the point with index %i.\n",
							name_.c_str(), index);
						continue;
					}

					omp_mem[tid][0] = e2c / e1c;
					omp_mem[tid][1] = e3c / e2c;;
					omp_mem[tid][2] = e3c;
				}

				for (int d = 0; d < omp_mem[tid].size(); d++)
					prg_mem[index][d] = omp_mem[tid][d];
			}

			for (index = 0; index < int(input_->size()); index++)
			{
				if (!borders[index])
				{
					if ((prg_mem[index][0] < gamma_21_) && (prg_mem[index][1] < gamma_32_))
						third_eigen_value_[index] = prg_mem[index][2];
				}
			}

			bool* feat_max = new bool[input_->size()];
			bool is_max;

#ifdef _OPENMP
#pragma omp parallel for private(is_max) num_threads(threads_)
#endif
			for (index = 0; index < int(input_->size()); index++)
			{
				feat_max[index] = false;
				PointInT current_point = input_->points[index];

				if ((third_eigen_value_[index] > 0.0) && (pcl::isFinite(current_point)))
				{
					std::vector<int> nn_indices;
					std::vector<float> nn_distances;
					int n_neighbors;

					this->searchForNeighbors(static_cast<int> (index), non_max_radius_, nn_indices, nn_distances);

					n_neighbors = static_cast<int> (nn_indices.size());

					if (n_neighbors >= min_neighbors_)
					{
						is_max = true;

						for (int j = 0; j < n_neighbors; j++)
							if (third_eigen_value_[index] < third_eigen_value_[nn_indices[j]])
								is_max = false;
						if (is_max)
							feat_max[index] = true;
					}
				}
			}

#ifdef _OPENMP
#pragma omp parallel for shared (output) num_threads(threads_)
#endif
			for (index = 0; index < int(input_->size()); index++)
			{
				if (feat_max[index])
#ifdef _OPENMP
#pragma omp critical
#endif
				{
					PointOutT p;
					p.getVector3fMap() = input_->points[index].getVector3fMap();
					output.points.push_back(p);
					keypoints_indices_->indices.push_back(index);
				}
			}

			output.header = input_->header;
			output.width = static_cast<uint32_t> (output.points.size());
			output.height = 1;

			// Clear the contents of variables and arrays before the beginning of the next computation.
			if (border_radius_ > 0.0)
				normals_.reset(new pcl::PointCloud<NormalT>);

			delete[] borders;
			delete[] prg_mem;
			delete[] prg_local_mem;
			delete[] feat_max;

		}


		/** \brief The radius of the spherical neighborhood used to compute the scatter matrix.*/
		double salient_radius_;

		/** \brief The non maxima suppression radius. */
		double non_max_radius_;

		/** \brief The radius used to compute the normals of the input cloud. */
		double normal_radius_;

		/** \brief The radius used to compute the boundary points of the input cloud. */
		double border_radius_;

		/** \brief The upper bound on the ratio between the second and the first eigenvalue returned by the EVD. */
		double gamma_21_;

		/** \brief The upper bound on the ratio between the third and the second eigenvalue returned by the EVD. */
		double gamma_32_;

		/** \brief Store the third eigen value associated to each point in the input cloud. */
		double *third_eigen_value_;

		/** \brief Store the information about the boundary points of the input cloud. */
		bool *edge_points_;

		/** \brief Minimum number of neighbors that has to be found while applying the non maxima suppression algorithm. */
		int min_neighbors_;

		/** \brief The cloud of normals related to the input surface. */
		PointCloudNConstPtr normals_;

		/** \brief The decision boundary (angle threshold) that marks points as boundary or regular. (default \f$\pi / 2.0\f$) */
		float angle_threshold_;

		/** \brief The number of threads that has to be used by the scheduler. */
		unsigned int threads_;

	};



}
int
main(int argc, char** argv)
{
	// Objects for storing the point cloud and the keypoints.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints2(new pcl::PointCloud<pcl::PointXYZ>);
	//For the mian dataset,computation tends to be quite expensive,so its better to subject
	//the scene cloud to downsampling and then perform actual computation on it.
	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) != 0)
	{
		return -1;
	}
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *cloud1) != 0)
	{
		return -1;
	}
	//std::vector<int> mapping;
	//pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
	//pcl::removeNaNFromPointCloud(*cloud1, *cloud1, mapping);
	
	//At this stage keypoint detection begins
	// ISS keypoint detector object.
// ISS keypoint detector object.
	pcl::ISSKeypoint3D1<pcl::PointXYZ, pcl::PointXYZ> detector;
	detector.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	detector.setSearchMethod(kdtree);
	double resolution = computeCloudResolution(cloud);
	// Set the radius of the spherical neighborhood used to compute the scatter matrix.
	detector.setSalientRadius(6 * resolution);
	// Set the radius for the application of the non maxima supression algorithm.
	detector.setNonMaxRadius(4 * resolution);
	// Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
	detector.setMinNeighbors(5);
	// Set the upper bound on the ratio between the second and the first eigenvalue.
	detector.setThreshold21(0.975);
	// Set the upper bound on the ratio between the third and the second eigenvalue.
	detector.setThreshold32(0.975);
	// Set the number of prpcessing threads to use. 0 sets it to automatic.
	detector.setNumberOfThreads(4);
	detector.compute(*keypoints);
detector.setInputCloud(cloud1);
detector.setSearchMethod(kdtree);
	double resolution1 = computeCloudResolution(cloud1);
	// Set the radius of the spherical neighborhood used to compute the scatter matrix.
	detector.setSalientRadius(6 * resolution1);
	// Set the radius for the application of the non maxima supression algorithm.
	detector.setNonMaxRadius(4 * resolution1);
	// Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
	detector.setMinNeighbors(5);
	// Set the upper bound on the ratio between the second and the first eigenvalue.
	detector.setThreshold21(0.975);
	// Set the upper bound on the ratio between the third and the second eigenvalue.
	detector.setThreshold32(0.975);
	// Set the number of prpcessing threads to use. 0 sets it to automatic.
	detector.setNumberOfThreads(4);
	detector.compute(*keypoints1);
	pcl::visualization::PCLVisualizer viewer1("PCL Viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler(keypoints1, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud1, 255, 0, 0);
	viewer1.setBackgroundColor(0.0, 0.0, 0.0);
	viewer1.addPointCloud(cloud, cloud_color_handler, "cloud");
	viewer1.addPointCloud(keypoints, keypoints_color_handler, "keypoints");
	viewer1.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");

	while (!viewer1.wasStopped())	{
	viewer1.spinOnce();
		}
	pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud);
	filter.setLeafSize(0.01f, 0.01f, 0.01f);

	filter.filter(*filteredCloud);
pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr normals1(new pcl::PointCloud<pcl::Normal>);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(0.03);
//	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);
	// PFH estimation object.
	// Object for storing the PFH descriptors for each point.
	//pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors1(new pcl::PointCloud<pcl::PFHSignature125>());
	//From this stage onwards feature descriptor and matching begins.
	// Estimate the normals.
	// PFH estimation object.
	
	


	// Display and retrieve the SHOT descriptor for the first point.
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors1(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> pfh;
	


	
	
	
	pfh.setInputCloud(cloud);
	pfh.setInputNormals(normals);
	pfh.setSearchMethod(kdtree);
	 //Search radius, to look for neighbors. Note: the value given here has to be
	// larger than the radius used to estimate the normals.
	//We're trying to see if FPH performs faster than PFH and if yes then are the results really consistent.
	pfh.setRadiusSearch(0.03);
	pfh.compute(*descriptors);
	
	std::cout << "imhere" << std::endl;
	//filtering is done to make the computation more efficient.
	filter.setInputCloud(cloud1);
	filter.setLeafSize(0.01f, 0.01f, 0.01f);

	filter.filter(*filteredCloud1);
	normalEstimation.setInputCloud(filteredCloud1);
	std::cout << "filtering done" << std::endl;
	normalEstimation.setRadiusSearch(0.03);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals1);
	std::cout << "normals found" << std::endl;
	pfh.setInputCloud(filteredCloud1);
	pfh.setInputNormals(normals1);
	pfh.setSearchMethod(kdtree);
	pfh.setRadiusSearch(0.05);
	std::cout << "initiating computation.Please wait this may take some time" << std::endl;
	pfh.compute(*descriptors1);
	std::cout << "descriptor estimation done" << std::endl;
	std::cout << descriptors->points.size()<<std::endl;
	std::cout << descriptors1->points.size();
	//Matching stage begins.
pcl::KdTreeFLANN<pcl::FPFHSignature33> matching;
	matching.setInputCloud(descriptors);
pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

	// Check every descriptor computed for the scene.
	for (size_t i = 0; i < descriptors1->size(); ++i)
	{
		std::vector<int> neighbors(1);
		std::vector<float> squaredDistances(1);
		
		
			// Find the nearest neighbor (in descriptor space)...
			int neighborCount = matching.nearestKSearch(descriptors1->at(i), 1, neighbors, squaredDistances);
			// ...and add a new correspondence if the distance is less than a threshold
			//for milk use 10 for everything else above 5000
			// (SHOT distances are between 0 and 1, other descriptors use different metrics).
			int ctr = 0;
			if (neighborCount == 1&&squaredDistances[0]<5000)
			{
				pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i), squaredDistances[0]);
				correspondences->push_back(correspondence);
				
			}
		
	}
	std::cout << "Found " << correspondences->size() << " correspondences." << std::endl;
	std::vector<pcl::Correspondences> clusteredCorrespondences;
	// Object for storing the transformations (rotation plus translation).
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;

pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> grouping;
	grouping.setSceneCloud(filteredCloud1);
	grouping.setInputCloud(filteredCloud);
	grouping.setModelSceneCorrespondences(correspondences);
	// Minimum cluster size. Default is 3 (as at least 3 correspondences
	// are needed to compute the 6 DoF pose).
	//earlier we set it to 9
	//for clorox 5000
	grouping.setGCThreshold(9);
	// Resolution of the consensus set used to cluster correspondences together,
	// in metric units. Default is 1.0.
	
	grouping.setGCSize(0.025);

	grouping.recognize(transformations, clusteredCorrespondences);
	
	std::cout << "Model instances found: " << transformations.size() << std::endl << std::endl;
	for (size_t i = 0; i < transformations.size(); i++)
	{
		std::cout << "Instance " << (i + 1) << ":" << std::endl;
		std::cout << "\tHas " << clusteredCorrespondences[i].size() << " correspondences." << std::endl << std::endl;

		Eigen::Matrix3f rotation = transformations[i].block<3, 3>(0, 0);
		Eigen::Vector3f translation = transformations[i].block<3, 1>(0, 3);
		printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
		printf("\t\tR = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
		printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
		std::cout << std::endl;
		printf("\t\tt = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
	}
	//Does the recognized instance align with the actual model instance,if not run ICP
	
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> registration;
	std::vector < pcl::PointCloud < pcl::PointXYZ > ::ConstPtr > instances;
	std::vector < pcl::PointCloud < pcl::PointXYZ > ::ConstPtr > registered_instances;
	for (size_t i = 0; i < transformations.size(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedcloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*cloud, *rotatedcloud, transformations[i]);
		instances.push_back(rotatedcloud);
	}
	for (size_t i = 0; i < transformations.size(); ++i)
	{
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(instances[i]);
		icp.setInputTarget(cloud);
		pcl::PointCloud<pcl::PointXYZ>::Ptr registered(new pcl::PointCloud<pcl::PointXYZ>);
		icp.align(*registered);
		registered_instances.push_back(registered);
		if (icp.hasConverged())
		{
			std::cout << "Converged. " << std::endl;

			Eigen::Matrix4f transformation = icp.getFinalTransformation();
			
		}
		else
			std::cout << "Not converged." << std::endl;
		
	}
	/*pcl::visualization::PCLVisualizer viewer1("Correspondence Grouping");
	viewer1.setCameraPosition(0.0, -1.0, -2.0, 0.0, 0.0, 1.0);
	viewer1.addPointCloud(filteredCloud1, "scene_cloud");
	pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*filteredCloud, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> off_scene_model_color_handler(off_scene_model_keypoints, 255, 255, 128);
	viewer1.addPointCloud(off_scene_model_keypoints, off_scene_model_color_handler, "off_scene_model");
	for (size_t i = 0; i < transformations.size(); ++i) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr rotmodel(new pcl::PointCloud<pcl::PointXYZ>());

		pcl::transformPointCloud(*filteredCloud, *rotmodel, transformations[i]);

		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rotmodel_color_handler(rotmodel, 255, 0, 0);
		viewer1.addPointCloud(rotmodel, rotmodel_color_handler, ss_cloud.str());


		for (size_t j = 0; j < clusteredCorrespondences[i].size(); ++j) {

			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			pcl::PointXYZ& model_point = off_scene_model_keypoints->at(clusteredCorrespondences[i][j].index_query);
			pcl::PointXYZ& scene_point = filteredCloud1->at(clusteredCorrespondences[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer1.addLine<pcl::PointXYZ, pcl::PointXYZ>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}
	}*/
	//Visualizing Results
	pcl::visualization::PCLVisualizer viewer("Hypotheses Verification");
	viewer.addPointCloud(filteredCloud1, "scene_cloud");
	for (size_t i = 0; i < instances.size(); ++i)
	{
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;

		CloudStyle clusterStyle = style_red;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> instance_color_handler(instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
		viewer.addPointCloud(instances[i], instance_color_handler, ss_instance.str());
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str());

		CloudStyle registeredStyles =  style_cyan;
		ss_instance << "_registered" << endl;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> registered_instance_color_handler(registered_instances[i], registeredStyles.r,
			registeredStyles.g, registeredStyles.b);
		viewer.addPointCloud(registered_instances[i], registered_instance_color_handler, ss_instance.str());
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, registeredStyles.size, ss_instance.str());
	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
