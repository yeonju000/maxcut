#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>
#include <chrono>
#include <set>

using namespace std;

class GA {
private:
	vector<string> population;
	vector<double> fitnesses; //적합도
	vector<vector<pair<int, int>>> graph;
	int populationSize;
	mt19937 gen;
	uniform_real_distribution<> dis;

public:
	GA(int populationSize, vector<vector<pair<int, int>>>& graph) : graph(graph), populationSize(populationSize), dis(0.0, 1.0) {
		random_device rd;
		gen = mt19937(rd());
	}

	struct Individual {
		std::string chromosome;
		int fitness;
		Individual(std::string chromo, int fit) : chromosome(chromo), fitness(fit) {}
	};

	//1. mergeSort
	void mergeSort(int left, int right) {
		if (left < right) {
			int mid = left + (right - left) / 2;
			mergeSort(left, mid);
			mergeSort(mid + 1, right);
			merge(left, mid, right);
		}
	}

	void merge(int left, int mid, int right) {
		vector<string> tempPopulation(right - left + 1);
		vector<double> tempFitnesses(right - left + 1);

		int i = left, j = mid + 1, k = 0;

		while (i <= mid && j <= right) {
			if (fitnesses[i] <= fitnesses[j]) {
				tempPopulation[k] = population[i];
				tempFitnesses[k] = fitnesses[i];
				++i;
			}
			else {
				tempPopulation[k] = population[j];
				tempFitnesses[k] = fitnesses[j];
				++j;
			}
			++k;
		}

		while (i <= mid) {
			tempPopulation[k] = population[i];
			tempFitnesses[k] = fitnesses[i];
			++i;
			++k;
		}

		while (j <= right) {
			tempPopulation[k] = population[j];
			tempFitnesses[k] = fitnesses[j];
			++j;
			++k;
		}

		for (i = left, k = 0; i <= right; ++i, ++k) {
			population[i] = tempPopulation[k];
			fitnesses[i] = tempFitnesses[k];
		}
	}


	//2. basicQuickSort
	void basicQuickSort(int low, int high) {
		if (low < high) {
			int pi = basicQuickPartition(low, high);

			basicQuickSort(low, pi - 1);
			basicQuickSort(pi + 1, high);
		}
	}

	//basicQuickSort의 partition
	int basicQuickPartition(int low, int high) {
		double pivot = fitnesses[low];
		int left = low + 1;
		int right = high;

		while (true) {
			while (left <= right && fitnesses[left] <= pivot) {
				left++;
			}
			while (left <= right && fitnesses[right] >= pivot) {
				right--;
			}
			if (left > right) {
				break;
			}
			swap(fitnesses[left], fitnesses[right]);
			swap(population[left], population[right]);
		}

		swap(fitnesses[low], fitnesses[right]);
		swap(population[low], population[right]);

		return right;
	}



	//3. intelligentquikSort
	void intelligentQuickSort(int low, int high) {
		if (low < high) {
			int pi = intelligentPartition(low, high);
			intelligentQuickSort(low, pi - 1);
			intelligentQuickSort(pi + 1, high);
		}
	}

	//Median of Medians
	double medianOfMedians(vector<double>& arr, int left, int right) {
		if (right - left + 1 <= 5) {
			sort(arr.begin() + left, arr.begin() + right + 1);
			return arr[left + (right - left) / 2];
		}

		vector<double> medians;
		int i;
		for (i = left; i <= right - 5; i += 5) {
			double median = findMedian(arr, i, 5);
			medians.push_back(median);
		}
		if (i <= right) {
			double median = findMedian(arr, i, right - i + 1);
			medians.push_back(median);
		}

		return medianOfMedians(medians, 0, medians.size() - 1);
	}

	double findMedian(vector<double>& arr, int start, int size) {
		sort(arr.begin() + start, arr.begin() + start + size);
		return arr[start + size / 2];
	}

	int intelligentPartition(int low, int high) {
		double pivot = medianOfMedians(fitnesses, low, high);
		int i = low;
		for (int j = low; j < high; j++) {
			if (fitnesses[j] < pivot) {
				swap(fitnesses[i], fitnesses[j]);
				swap(population[i], population[j]);
				i++;
			}
		}
		swap(fitnesses[i], fitnesses[high]);
		swap(population[i], population[high]);
		return i;
	}

	//4. paranoidquickSort
	void paranoidQuickSort(int low, int high) {
		if (low < high) {
			int pi = paranoidPartition(low, high);
			paranoidQuickSort(low, pi - 1);
			paranoidQuickSort(pi + 1, high);
		}
	}


	//5. countingSort
	void countingSort(vector<double>& fitnesses, vector<string>& population) {
		int populationSize = fitnesses.size();
		double maxFitness = *max_element(fitnesses.begin(), fitnesses.end());
		double minFitness = *min_element(fitnesses.begin(), fitnesses.end());

		int indexRange = static_cast<int>(ceil((maxFitness - minFitness)) + 1);

		//인덱스별로 저장할 버킷
		vector<vector<int>> buckets(indexRange);

		//각 개체를 버켓에 놓는다
		for (int i = 0; i < populationSize; ++i) {
			int index = static_cast<int>((fitnesses[i] - minFitness));
			buckets[index].push_back(i);
		}

		//개체 다시 정렬
		vector<string> sortedPopulation(populationSize);
		vector<double> sortedFitnesses(populationSize);
		int idx = 0;
		for (size_t i = 0; i < buckets.size(); ++i) {
			for (size_t j = 0; j < buckets[i].size(); ++j) {
				sortedPopulation[idx] = population[buckets[i][j]];
				sortedFitnesses[idx] = fitnesses[buckets[i][j]];
				idx++;
			}
		}

		population = sortedPopulation;
		fitnesses = sortedFitnesses;
	}


	//pranoid partition
	int paranoidPartition(int low, int high) {
		int n = high - low + 1;
		int pivotIndex = low + n / 4 + (n / 2 - n / 4) * (gen() % 2); // 1/4 또는 3/4 위치 선택
		double pivotValue = fitnesses[pivotIndex];

		swap(fitnesses[pivotIndex], fitnesses[high]);
		swap(population[pivotIndex], population[high]);

		int i = low;
		for (int j = low; j < high; j++) {
			if (fitnesses[j] < pivotValue) {
				swap(fitnesses[i], fitnesses[j]);
				swap(population[i], population[j]);
				i++;
			}
		}

		swap(fitnesses[i], fitnesses[high]);
		swap(population[i], population[high]);
		return i;
	}


	//초기화
	void initializePopulation() {
		for (int i = 0; i < populationSize; ++i) {
			string chromosome; //크로모종
			for (size_t j = 0; j < graph.size(); ++j) {
				chromosome += dis(gen) < 0.5 ? '0' : '1';
			}
			population.push_back(chromosome);
			fitnesses.push_back(evaluateFitness(chromosome));
		}
	}

	//적합도 평가
	//모든 정점을 돌면서 선택된 정점과 선택되지 않은 정점 가중치를 더하기
	//i번째 정점이 선택됐고 이것과 연결된 정점이 선택되지 않았다면 가중치를 더함
	double evaluateFitness(const string& solution) {
		double totalWeight = 0;

		for (int i = 0; i < static_cast<int>(graph.size()); ++i) {
			for (const auto& edge : graph[i]) {
				if (solution[i] == '1' && solution[edge.first] == '0') {
					totalWeight += edge.second;
				}
			}
		}
		return totalWeight;
	}

	/*
	//룰렛휠
	string Selection() {
		double totalFitness = accumulate(fitnesses.begin(), fitnesses.end(), 0.0);

		//cost 합이 0인 경우
		if (totalFitness <= 0) {
			return population[rand() % populationSize];
		}

		double slice = dis(gen) * totalFitness;
		double sum = 0.0;
		for (int i = 0; i < populationSize; ++i) {
			sum += fitnesses[i];
			if (sum >= slice) {
				return population[i];
			}
		}

		//예방...
		return population.back();
	}
	*/

	//토너먼트
	string Selection() {
		uniform_int_distribution<> dist(0, populationSize - 1);
		int idx1 = dist(gen);
		int idx2 = dist(gen);
		while (idx2 == idx1) {
			idx2 = dist(gen);
		}

		double fitness1 = fitnesses[idx1];
		double fitness2 = fitnesses[idx2];

		//cost가 더 좋은 애가 선택될 확률을 60%
		double winProb = 0.6;

		bool isFirstFitter = fitness1 > fitness2;
		double fitterFitness = isFirstFitter ? fitness1 : fitness2;
		double lesserFitness = isFirstFitter ? fitness2 : fitness1;

		if (fitterFitness > lesserFitness) {
			if (dis(gen) < winProb) {
				return population[isFirstFitter ? idx1 : idx2];
			}
			else {
				return population[isFirstFitter ? idx2 : idx1];
			}
		}
		else {
			//만약 두 적합도가 같으면 랜덤으로 하나를 선택
			return dis(gen) < 0.5 ? population[idx1] : population[idx2];
		}
	}

	/*
	//crossover 원포인트
	string crossover(const string& parent1, const string& parent2) {
		int midPoint = parent1.size() / 2; //부모 mid 계산
		return parent1.substr(0, midPoint) + parent2.substr(midPoint);
	}
	*/

	/*
	//crossover 투포인트
	string crossover(const string& parent1, const string& parent2) {
	  int length = parent1.size();
	  uniform_int_distribution<> dist(0, length - 1);
	  int point1 = dist(gen);
	  int point2 = dist(gen);

	  //point1이 항상 point2보다 작거나 같게 만듦
	  if (point1 > point2) swap(point1, point2);

	  string offspring = parent1;
	  for (int i = point1; i <= point2; ++i) {
		  offspring[i] = parent2[i];
	  }

	  return offspring;
	}
  */


    //Uniform Crossover
	string crossover(const string& parent1, const string& parent2) {
		string offspring = "";
		for (size_t i = 0; i < parent1.size(); ++i) {
			offspring += dis(gen) < 0.7 ? parent1[i] : parent2[i];
		}
		return offspring;
	}
	//숫자 올리니까 성능 높아짐


    //뮤테이션
	string mutation(const string& chromosome) {
		double mutationRate = pow(10, -2); //10의 -2승
		string mutatedChromosome = chromosome;
		for (size_t i = 0; i < chromosome.size(); ++i) {
			if (dis(gen) < mutationRate) {
				mutatedChromosome[i] = mutatedChromosome[i] == '0' ? '1' : '0';
			}
		}
		return mutatedChromosome;
	}

	void replacement(const string& offspring, double offspringFitness) {
		int numToReplace = 150; //교체할 개체 수
		int replaced = 0;

		//적합도가 낮은 순서대로 교체
		for (int i = 0; i < populationSize && replaced < numToReplace; ++i) {
			if (fitnesses[i] < offspringFitness) {
				population[i] = offspring;
				fitnesses[i] = offspringFitness;
				replaced++;
			}
		}
	}

	void run(int generations) {
		auto startTime = chrono::steady_clock::now();
		for (int i = 0; i < generations; ++i) {
			auto currentTime = chrono::steady_clock::now();
			auto elapsedTime = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
			if (elapsedTime > 179.9) {
				cout << "Time limit!" << endl;
				break;
			}

			for (int j = 0; j < populationSize; ++j) {
				string parent1 = Selection();
				string parent2 = Selection();
				string offspring = crossover(parent1, parent2);
				offspring = mutation(offspring);
				double offspringFitness = evaluateFitness(offspring);

				//교체
				replacement(offspring, offspringFitness);

			}

			//mergeSort(0, populationSize - 1);
			//basicQuickSort(0, populationSize - 1);
			intelligentQuickSort(0, populationSize - 1);
			//paranoidQuickSort(0, fitnesses.size() - 1);
			//countingSort(fitnesses, population);
		}
	}


	void outputResult(const string& filename) {
		ofstream outFile(filename);
		auto maxIt = max_element(fitnesses.begin(), fitnesses.end());
		size_t index = distance(fitnesses.begin(), maxIt);
		for (size_t i = 0; i < population[index].size(); ++i) {
			if (population[index][i] == '1') {
				outFile << i + 1 << " ";
			}
		}

		outFile << "\n";
	}
};


int main() {
	ifstream inputFile("maxcut.in");
	ofstream outputFile("maxcut.out");
	int V, E;
	inputFile >> V >> E;
	vector<vector<pair<int, int>>> graph(V);
	for (int i = 0; i < E; ++i) {
		int from, to, weight;
		inputFile >> from >> to >> weight;
		graph[from - 1].emplace_back(to - 1, weight);
		graph[to - 1].emplace_back(from - 1, weight);
	}
	int populationSize = 500;
	int generations = 1000;
	GA ga(populationSize, graph);
	ga.initializePopulation();
	ga.run(generations);
	ga.outputResult("maxcut.out");

	return 0;
}
