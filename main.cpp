#include "scatterdatamodifier.h"

#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFontComboBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMessageBox>
#include <QtGui/QScreen>
#include <QtGui/QFontDatabase>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <iostream>

Eigen::Vector3d extrinsicTranslation;
Eigen::Matrix3d extrinsicRotation;

Eigen::MatrixX3d
transformLiDARDataToIMUFrame(const std::vector<std::vector<double>>& data) {
    Eigen::MatrixX3d points(data.size(), 3);

    long int row = 0;
    for (auto it = data.begin(); it != data.end(); ++it, ++row) {
        points(row, 0) = (*it)[0];
        points(row, 1) = (*it)[1];
        points(row, 2) = (*it)[2];
    }

    return (extrinsicRotation * points.transpose()).transpose().rowwise() + extrinsicTranslation.transpose();
}

class GroundTruth {
public:
    GroundTruth() = default;

    GroundTruth(const GroundTruth& other) :
      acceleration(other.acceleration), velocity(other.velocity), position(other.position),
      angularAcceleration(other.angularAcceleration), angularVelocity(other.angularVelocity), distance(other.distance) {
    }

    GroundTruth& operator=(const GroundTruth& other) {
        if (this == &other)
            return *this;
        acceleration = other.acceleration;
        velocity = other.velocity;
        position = other.position;
        angularAcceleration = other.angularAcceleration;
        angularVelocity = other.angularVelocity;
        distance = other.distance;
        return *this;
    }

    std::vector<std::vector<double>>& acceleration1() { return acceleration; }
    std::vector<std::vector<double>> velocity1() const { return velocity; }
    std::vector<std::vector<double>> distance1() const { return distance; }

    const std::vector<std::vector<double>>& getPosition() const { return position; }

    void set_acceleration(std::vector<std::vector<double>>&& acceleration) {
        this->acceleration = std::move(acceleration);
    }

    void set_velocity(std::vector<std::vector<double>>&& velocity) { this->velocity = std::move(velocity); }
    void set_position(std::vector<std::vector<double>>&& position) { this->position = std::move(position); }

    void set_angular_acceleration(std::vector<std::vector<double>>&& angular_acceleration) {
        angularAcceleration = std::move(angular_acceleration);
    }

    void set_angular_velocity(std::vector<std::vector<double>>&& angular_velocity) {
        angularVelocity = std::move(angular_velocity);
    }

    void set_distance(std::vector<std::vector<double>>&& distance) { this->distance = std::move(distance); }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & acceleration;
        ar & velocity;
        ar & position;

        ar & angularAcceleration;
        ar & angularVelocity;
        ar & distance;
    }

    std::vector<std::vector<double>> acceleration, velocity, position;
    std::vector<std::vector<double>> angularAcceleration, angularVelocity, distance;
};

class SensorData {
public:
    SensorData() = default;

    SensorData(std::vector<std::vector<double>>&& data, std::vector<double>&& timestamp) :
      data(std::move(data)), timestamp(std::move(timestamp)) {}

    void set_data(std::vector<std::vector<double>>&& data) { this->data = std::move(data); }
    void set_timestamp(std::vector<double>&& timestamp) { this->timestamp = std::move(timestamp); }
    std::vector<std::vector<double>>& data1() { return data; }
    std::vector<double>& timestamp1() { return timestamp; }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & data;
        ar & timestamp;
    }

    std::vector<std::vector<double>> data;
    std::vector<double> timestamp;
};

class IMUMeasurement {
public:
    IMUMeasurement() = default;

    IMUMeasurement(const SensorData& acceleration, const SensorData& angular_velocity) :
      acceleration(acceleration), angularVelocity(angular_velocity) {}
    SensorData acceleration1() const { return acceleration; }
    SensorData angular_velocity() const { return angularVelocity; }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & acceleration;
        ar & angularVelocity;
    }

    SensorData acceleration, angularVelocity;
};

class Data {
public:
    Data() = default;

    Data(const GroundTruth& ground_truth,
         const IMUMeasurement& imu_measurements,
         const SensorData& gnss_measurement,
         const SensorData& li_dar_measurement) :
      groundTruth(ground_truth), IMUMeasurements(imu_measurements), GNSSMeasurement(gnss_measurement),
      LiDARMeasurement(li_dar_measurement) {}

    Data(Data&& other) :
      groundTruth(std::move(other.groundTruth)), IMUMeasurements(std::move(other.IMUMeasurements)),
      GNSSMeasurement(std::move(other.GNSSMeasurement)), LiDARMeasurement(std::move(other.LiDARMeasurement)) {}

    Data& operator=(Data&& other) {
        if (this == &other)
            return *this;
        groundTruth = std::move(other.groundTruth);
        IMUMeasurements = std::move(other.IMUMeasurements);
        GNSSMeasurement = std::move(other.GNSSMeasurement);
        LiDARMeasurement = std::move(other.LiDARMeasurement);
        return *this;
    }

    GroundTruth& ground_truth() { return groundTruth; }
    IMUMeasurement& imu_measurements() { return IMUMeasurements; }
    SensorData& gnss_measurement() { return GNSSMeasurement; }
    SensorData& li_dar_measurement() { return LiDARMeasurement; }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & groundTruth;
        ar & IMUMeasurements;
        ar & GNSSMeasurement;
        ar & LiDARMeasurement;
    }

    GroundTruth groundTruth;
    IMUMeasurement IMUMeasurements;
    SensorData GNSSMeasurement, LiDARMeasurement;
};


Eigen::Quaterniond eulerToQuaternion(const std::vector<double> &euler) {

    Eigen::AngleAxisd roll(euler[0], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(euler[1], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(euler[2], Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yaw * pitch * roll;
    return q;

}




int main(int argc, char **argv)
{
    //! [0]
    QApplication app(argc, argv);

    Data newg;
    {
        std::ifstream ifs("mydata");
        boost::archive::text_iarchive ia(ifs);

        ia >> newg;
    }


    extrinsicTranslation << 0.5, 0.1, 0.5;
    extrinsicRotation << 0.99376, -0.09722, 0.05466, 0.09971, 0.99401, -0.04475, -0.04998, 0.04992, 0.9975;

    Eigen::MatrixX3d LiDAR = transformLiDARDataToIMUFrame(newg.li_dar_measurement().data1());
    // std::cout << LiDAR << std::endl;

    double varianceIMUF = 0.1;
    double varianceIMUW = 0.25;
    double varianceGNSS = 10.0;
    double varianceLiDAR = 10.0;

    Eigen::Vector3d gravity;
    gravity << 0, 0, -9.81;

    Eigen::MatrixXd lJacobian = Eigen::MatrixXd::Zero(9, 6);
    Eigen::MatrixXd hJacobian = Eigen::MatrixXd::Zero(3, 9);

    lJacobian.block<6, 6>(3, 0) = Eigen::Matrix<double, 6, 6>::Identity();
    hJacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    // std::cout << lJacobian << std::endl;
    // std::vector<std::vector<double>> aaa = newg.imu_measurements().acceleration1().data1();
    // std::cout << aaa.size() << std::endl;

    // Eigen::MatrixX3d positionEstimates(newg.imu_measurements().acceleration1().data1().size(), 3);
    // Eigen::MatrixX3d velocityEstimates(newg.imu_measurements().acceleration1().data1().size(), 3);
    std::vector<Eigen::Vector3d> positionEstimates(newg.imu_measurements().acceleration1().data1().size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> velocityEstimates(newg.imu_measurements().acceleration1().data1().size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Quaterniond> orientationEstimates(newg.imu_measurements().acceleration1().data1().size(), Eigen::Quaterniond::Identity());
    // Eigen::MatrixXd orientationEstimatesAsQuaternions(newg.imu_measurements().acceleration1().data1().size(), 4);

    // Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    /*
    for (long int i = 0; i < newg.imu_measurements().acceleration1().data1().size(); ++i)
    {
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        orientationEstimatesAsQuaternions.row(i) << q.w(), q.vec().transpose();
    }
    */
    // std::cout << q.w() << std::endl;

    std::vector<Eigen::MatrixXd> covarianceMatrices(newg.imu_measurements().acceleration1().data1().size(), Eigen::MatrixXd::Zero(9, 9));

    // std::cout << covarianceMatrices[0] << std::endl;

    std::vector<std::vector<double>> position = newg.ground_truth().getPosition();

    positionEstimates[0] = Eigen::Vector3d(position[0][0], position[0][1], position[0][2]);
    velocityEstimates[0] = Eigen::Vector3d(newg.ground_truth().velocity1()[0][0], newg.ground_truth().velocity1()[0][1], newg.ground_truth().velocity1()[0][2]);

    // std::cout << velocityEstimates[0] << std::endl;

    orientationEstimates[0] = eulerToQuaternion(newg.ground_truth().distance1()[0]);
    // std::cout << orientationEstimates[1] << std::endl;
    Eigen::Matrix3d cNS0 = orientationEstimates[0].normalized().toRotationMatrix();
    // std::cout << cNS0 << std::endl;


    Eigen::Matrix3d RGNSS = Eigen::Matrix3d::Identity() * varianceGNSS;
    Eigen::Matrix3d RLiDAR = Eigen::Matrix3d::Identity() * varianceLiDAR;

    Eigen::Matrix<double, 9, 9> fK = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 6> lK = Eigen::Matrix<double, 9, 6>::Zero();
    lK.block<6, 6>(3, 0) = Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
    Q.block<3, 3>(0, 0) *= varianceIMUF;
    Q.block<3, 3>(3, 3) *= varianceIMUW;

    std::vector<double> timeIMUF = newg.imu_measurements().acceleration1().timestamp1();
    std::vector<double> timeGNSS = newg.gnss_measurement().timestamp1();
    std::vector<double> timeLiDAR = newg.li_dar_measurement().timestamp1();
    /*
    unsigned int mydatasize = newg.imu_measurements().acceleration1().data1().size();

    std::vector<Eigen::Vector3d> IMUFdatacik(mydatasize, Eigen::Vector3d::Zero());

    for (size_t i = 0; i < mydatasize; ++i) {
        IMUFdatacik[i] = Eigen::Vector3d(newg.imu_measurements().acceleration1().data1()[i][0], newg.imu_measurements().acceleration1().data1()[i][1], newg.imu_measurements().acceleration1().data1()[i][2]);
    }
    std::cout << IMUFdatacik[1] << std::endl;
    */
    unsigned int mydatasize = newg.imu_measurements().acceleration1().data1().size();
    for (int k = 1; k < mydatasize; ++k)
    {
        double deltaTime = timeIMUF[k] - timeIMUF[k-1];
        Eigen::Matrix<double, 6, 6> Qk = Q * deltaTime * deltaTime;

        Eigen::Matrix3d cns = orientationEstimates[k-1].normalized().toRotationMatrix();
        Eigen::Vector3d IMUFdatacik = Eigen::Vector3d(newg.imu_measurements().acceleration1().data1()[k-1][0], newg.imu_measurements().acceleration1().data1()[k-1][1], newg.imu_measurements().acceleration1().data1()[k-1][2]);
        positionEstimates[k] = positionEstimates[k-1] + deltaTime * velocityEstimates[k-1] + 0.5 * deltaTime * deltaTime * (cns * IMUFdatacik + gravity);
    }




















































































































































































































    Q3DScatter *graph = new Q3DScatter();
    QWidget *container = QWidget::createWindowContainer(graph);
    //! [0]

    if (!graph->hasContext()) {
        QMessageBox msgBox;
        msgBox.setText("Couldn't initialize the OpenGL context.");
        msgBox.exec();
        return -1;
    }

    QSize screenSize = graph->screen()->size();
    container->setMinimumSize(QSize(screenSize.width() / 2, screenSize.height() / 1.5));
    container->setMaximumSize(screenSize);
    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    container->setFocusPolicy(Qt::StrongFocus);

    //! [1]
    QWidget *widget = new QWidget;
    QHBoxLayout *hLayout = new QHBoxLayout(widget);
    QVBoxLayout *vLayout = new QVBoxLayout();
    hLayout->addWidget(container, 1);
    hLayout->addLayout(vLayout);
    //! [1]

    widget->setWindowTitle(QStringLiteral("A Cosine Wave"));

    //! [4]
    QComboBox *themeList = new QComboBox(widget);
    themeList->addItem(QStringLiteral("Qt"));
    themeList->addItem(QStringLiteral("Primary Colors"));
    themeList->addItem(QStringLiteral("Digia"));
    themeList->addItem(QStringLiteral("Stone Moss"));
    themeList->addItem(QStringLiteral("Army Blue"));
    themeList->addItem(QStringLiteral("Retro"));
    themeList->addItem(QStringLiteral("Ebony"));
    themeList->addItem(QStringLiteral("Isabelle"));
    themeList->setCurrentIndex(6);

    QPushButton *labelButton = new QPushButton(widget);
    labelButton->setText(QStringLiteral("Change label style"));

    QCheckBox *smoothCheckBox = new QCheckBox(widget);
    smoothCheckBox->setText(QStringLiteral("Smooth dots"));
    smoothCheckBox->setChecked(true);

    QComboBox *itemStyleList = new QComboBox(widget);
    itemStyleList->addItem(QStringLiteral("Sphere"), int(QAbstract3DSeries::MeshSphere));
    itemStyleList->addItem(QStringLiteral("Cube"), int(QAbstract3DSeries::MeshCube));
    itemStyleList->addItem(QStringLiteral("Minimal"), int(QAbstract3DSeries::MeshMinimal));
    itemStyleList->addItem(QStringLiteral("Point"), int(QAbstract3DSeries::MeshPoint));
    itemStyleList->setCurrentIndex(0);

    QPushButton *cameraButton = new QPushButton(widget);
    cameraButton->setText(QStringLiteral("Change camera preset"));

    QPushButton *itemCountButton = new QPushButton(widget);
    itemCountButton->setText(QStringLiteral("Toggle item count"));

    QCheckBox *backgroundCheckBox = new QCheckBox(widget);
    backgroundCheckBox->setText(QStringLiteral("Show background"));
    backgroundCheckBox->setChecked(true);

    QCheckBox *gridCheckBox = new QCheckBox(widget);
    gridCheckBox->setText(QStringLiteral("Show grid"));
    gridCheckBox->setChecked(true);

    QComboBox *shadowQuality = new QComboBox(widget);
    shadowQuality->addItem(QStringLiteral("None"));
    shadowQuality->addItem(QStringLiteral("Low"));
    shadowQuality->addItem(QStringLiteral("Medium"));
    shadowQuality->addItem(QStringLiteral("High"));
    shadowQuality->addItem(QStringLiteral("Low Soft"));
    shadowQuality->addItem(QStringLiteral("Medium Soft"));
    shadowQuality->addItem(QStringLiteral("High Soft"));
    shadowQuality->setCurrentIndex(4);

    QFontComboBox *fontList = new QFontComboBox(widget);
    fontList->setCurrentFont(QFont("Arial"));
    //! [4]

    //! [5]
    vLayout->addWidget(labelButton, 0, Qt::AlignTop);
    vLayout->addWidget(cameraButton, 0, Qt::AlignTop);
    vLayout->addWidget(itemCountButton, 0, Qt::AlignTop);
    vLayout->addWidget(backgroundCheckBox);
    vLayout->addWidget(gridCheckBox);
    vLayout->addWidget(smoothCheckBox, 0, Qt::AlignTop);
    vLayout->addWidget(new QLabel(QStringLiteral("Change dot style")));
    vLayout->addWidget(itemStyleList);
    vLayout->addWidget(new QLabel(QStringLiteral("Change theme")));
    vLayout->addWidget(themeList);
    vLayout->addWidget(new QLabel(QStringLiteral("Adjust shadow quality")));
    vLayout->addWidget(shadowQuality);
    vLayout->addWidget(new QLabel(QStringLiteral("Change font")));
    vLayout->addWidget(fontList, 1, Qt::AlignTop);
    //! [5]

    //! [2]
    ScatterDataModifier *modifier = new ScatterDataModifier(graph, position);
    //! [2]

    //! [6]
    QObject::connect(cameraButton, &QPushButton::clicked, modifier,
                     &ScatterDataModifier::changePresetCamera);
    QObject::connect(labelButton, &QPushButton::clicked, modifier,
                     &ScatterDataModifier::changeLabelStyle);
    QObject::connect(itemCountButton, &QPushButton::clicked, modifier,
                     &ScatterDataModifier::toggleItemCount);

    QObject::connect(backgroundCheckBox, &QCheckBox::stateChanged, modifier,
                     &ScatterDataModifier::setBackgroundEnabled);
    QObject::connect(gridCheckBox, &QCheckBox::stateChanged, modifier,
                     &ScatterDataModifier::setGridEnabled);
    QObject::connect(smoothCheckBox, &QCheckBox::stateChanged, modifier,
                     &ScatterDataModifier::setSmoothDots);

    QObject::connect(modifier, &ScatterDataModifier::backgroundEnabledChanged,
                     backgroundCheckBox, &QCheckBox::setChecked);
    QObject::connect(modifier, &ScatterDataModifier::gridEnabledChanged,
                     gridCheckBox, &QCheckBox::setChecked);
    QObject::connect(itemStyleList, SIGNAL(currentIndexChanged(int)), modifier,
                     SLOT(changeStyle(int)));

    QObject::connect(themeList, SIGNAL(currentIndexChanged(int)), modifier,
                     SLOT(changeTheme(int)));

    QObject::connect(shadowQuality, SIGNAL(currentIndexChanged(int)), modifier,
                     SLOT(changeShadowQuality(int)));

    QObject::connect(modifier, &ScatterDataModifier::shadowQualityChanged, shadowQuality,
                     &QComboBox::setCurrentIndex);
    QObject::connect(graph, &Q3DScatter::shadowQualityChanged, modifier,
                     &ScatterDataModifier::shadowQualityUpdatedByVisual);

    QObject::connect(fontList, &QFontComboBox::currentFontChanged, modifier,
                     &ScatterDataModifier::changeFont);

    QObject::connect(modifier, &ScatterDataModifier::fontChanged, fontList,
                     &QFontComboBox::setCurrentFont);
    //! [6]

    //! [3]
    widget->show();
    return app.exec();
    //! [3]
}
