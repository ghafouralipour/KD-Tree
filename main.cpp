#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

/* In machine learning, we sometimes have to find the nearest neighbour for a
 * given point. This can be efficiently done using k-d trees. 
 * Compile with g++ -std=c++17 task2.cpp -o task2 */

constexpr int pointDims = 3;
typedef std::array<float, pointDims> Point;

struct Node {
	std::shared_ptr<Node> left;
	std::shared_ptr<Node> right;
	Point point;

	Node(Point& pt) : left(nullptr), right(nullptr), point(pt) { }
};
float dist(Point p0, Point p1) {
	float distance = 0.0;
	for (int i = 0; i < pointDims; ++i) {
		distance += std::pow(p0[i] - p1[i], 2.0);
	}
	return std::pow(distance, 0.5);
}

std::shared_ptr<Node> closest(
	std::shared_ptr<Node> n0,
	std::shared_ptr<Node> n1,
	Point queryPoint) {

	if (n0 == nullptr) return n1;

	if (n1 == nullptr) return n0;

	float d1 = dist(n0->point, queryPoint);
	float d2 = dist(n1->point, queryPoint);

	if (d1 < d2)
		return n0;
	else
		return n1;
}
float distSquared(Point p0, Point p1) {
	long total = 0;
	int numDims = p0.size();

	for (int i = 0; i < numDims; i++) {
		int diff = std::abs(p0[i] - p1[i]);
		total += std::pow(diff, 2);
	}
	return total;
}
class KDTree {
public:
	KDTree(std::vector<Point>& points) {
		root_ = constructTree(points, 0);
	}

	Point nearestNeighbour(Point queryPoint) const {
		//auto [node, dist] = nearestPointInSubtree(queryPoint, root_, 0);
		return nearestPointInSubtree(queryPoint, root_, 0)->point;
	}

private:
	std::shared_ptr<Node> constructTree(std::vector<Point>& points, int depth) {
		if (points.size() == 0) {
			return nullptr;
		}

		int axis = depth % pointDims;
		std::sort(points.begin(), points.end(),
			[&](const Point & a, const Point & b) {
				return a[axis] < b[axis];
			});
		int medianIdx = points.size() / 2;
		auto node = std::make_shared<Node>(points[medianIdx]);
		std::vector<Point> leftPts(medianIdx);
		std::copy(points.begin(), points.begin() + medianIdx, leftPts.begin());
		std::vector<Point> rightPts(points.size() - medianIdx - 1);
		std::copy(points.begin() + medianIdx + 1, points.end(), rightPts.begin());
		node->left = constructTree(leftPts, depth + 1);
		node->right = constructTree(rightPts, depth + 1);
		return node;
	}
		
	std::shared_ptr<Node> nearestPointInSubtree(
		Point queryPoint, std::shared_ptr<Node> rootNode, int depth) const {

		int axis = depth % pointDims;
		if (rootNode == nullptr) return NULL;
		
		
		std::shared_ptr<Node>  nextBranch;
		std::shared_ptr<Node>  otherBranch;
		// compare the property appropriate for the current depth
		if (queryPoint[axis] < rootNode->point[axis]) {
			nextBranch = rootNode->left;
			otherBranch = rootNode->right;
		}
		else {
			nextBranch = rootNode->right;
			otherBranch = rootNode->left;
		}
		std::shared_ptr<Node> temp = nearestPointInSubtree(queryPoint, nextBranch, depth + 1);
		std::shared_ptr<Node> best = closest(temp, rootNode, queryPoint);
		
		float radiusSquared = distSquared(queryPoint, best->point);
		
		float axisDistToRoot = queryPoint[axis] - rootNode->point[axis];

		if (radiusSquared < axisDistToRoot * axisDistToRoot) {
			temp = nearestPointInSubtree(queryPoint,otherBranch, depth + 1);
			best = closest(temp, best, queryPoint);
		}

		return best;		
	}
	std::shared_ptr<Node> root_;
};

void printPoint(Point& point) {
	std::cout << "(";
	for (int i = 0; i < pointDims; ++i) {
		std::cout << point[i];
		if (i + 1 < pointDims) {
			std::cout << ", ";
		}
	}
	std::cout << ")" << std::endl << std::endl;
};

int main() {
	// Insert some points to the tree
	std::vector<Point> points{
		{0.935, 0.086, 0.069},
		{0.407, 0.5  , 0.349},
		{0.959, 0.394, 0.004},
		{0.418, 0.608, 0.452},
		{0.331, 0.704, 0.418},
		{0.76 , 0.988, 0.544},
		{0.89 , 0.063, 0.137},
		{0.574, 0.903, 0.101},
		{0.9  , 0.889, 0.708},
		{0.322, 0.963, 0.816}
	};
	KDTree tree(points);

	// Find nearest neighbour to a given point
	Point nn = tree.nearestNeighbour({ 0.5, 0.5, 0.5 });
	std::cout << "I found the following nearest neighbour for (0.5, 0.5, 0.5),"
		<< std::endl << "[expected (0.418, 0.608, 0.452)]:" << std::endl;
	printPoint(nn);

	nn = tree.nearestNeighbour({ 0.2, 0.7, 0.8 });
	std::cout << "I found the following nearest neighbour for (0.2, 0.7, 0.8),"
		<< std::endl << " [expected (0.322, 0.963, 0.816)]:" << std::endl;
	printPoint(nn);

	nn = tree.nearestNeighbour({ 0.7, 0.2, 0.5 });
	std::cout << "I found the following nearest neighbour for (0.7, 0.2, 0.5),"
		<< std::endl << " [expected (0.89 , 0.063, 0.137)]:" << std::endl;
	printPoint(nn);

	return 0;
}
