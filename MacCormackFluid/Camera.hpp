#pragma once

#include "StdAfx.hpp"

class Camera
{
private:
	glm::mat4 ViewMatrix;
	glm::mat4 ProjectionMatrix;


	// Initial position : on +Z
	glm::vec3 position = glm::vec3(0, -5,0);
	// Initial horizontal angle : toward -Z
	float horizontalAngle = 3.14f;
	// Initial vertical angle : none
	float verticalAngle = 0.0f;
	// Initial Field of View
	float initialFoV = 45.0f;

	float speed = 3.0f; // 3 units / second
	float mouseSpeed = 0.005f;

	GLFWwindow* window;


	int windowWidth;
	int windowHeigt;

	void Camera::ScrollCallback(GLFWwindow* window, double x, double y);

public:
	Camera(GLFWwindow* window);
	~Camera();

	void computeMatricesFromInputs();
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();
};

