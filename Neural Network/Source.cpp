#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <time.h>
#include <conio.h>
#include <SFML/Graphics.hpp>
using namespace std;

long double sigmoid(long double arg)
{
	return 1 / (1 + exp(-arg));
}

class Neuron
{
public:
	long double b;
	vector<long double> w;
	void init(int flows)
	{
		w.resize(flows);
		for (int i = 0; i < flows; i++)
			w[i] = (long double)(rand() % 10000) / 10000;
		b = (long double)(rand() % 10000) / 10000;
	}
};

class Layer
{
public:
	int inputs;
	int size;
	vector<long double> a, e;
	vector<Neuron> nr;
	void init(int neurons, int flows)
	{
		size = neurons;
		inputs = flows;
		a.resize(size);
		e.resize(size);
		nr.resize(size);
		for (int i = 0; i < size; i++)
			nr[i].init(inputs);
	}
	void calculate(vector<long double> in)
	{
		for (int i = 0; i < size; i++)
		{
			long double sum = 0;
			for (int j = 0; j < inputs; j++)
				sum += nr[i].w[j] * in[j];
			a[i] = sigmoid(sum + nr[i].b);
			e[i] = exp(-sum - nr[i].b);
			if (isinf(e[i]))
			{
				e[i] = numeric_limits<long double>::max();
			}
		}
	}
};

class Input
{
public:
	int size;
	vector<long double> x;
	long double y;
	void init(int flows)
	{
		size = flows;
		x.resize(size);
		for (int i = 0; i < 2; i++)
			x[i] = (long double)(rand() % 495 - 247) / 100;
		x[2] = x[0] * x[0];
		x[3] = x[1] * x[1];
		x[4] = sin(x[0] * 100);
		x[5] = sin(x[1] * 100);
		x[6] = x[0] * x[1];
		y = ((x[1] > 1 / x[0] && x[0] > 0) || (x[1] < 1 / x[0] && x[0] < 0)) ? 1 : 0;
	}
};

int main()
{
	srand(time(0));
	const long double ETA = 0.1, ACCURACY = 0.001;
	const int POOL_SIZE = 20000, BATCH_SIZE = 100, FLOW_SIZE = 7;
	vector<Input> pool(POOL_SIZE);
	for (int i = 0; i < POOL_SIZE; i++)
		pool[i].init(FLOW_SIZE);

	const int N_LAYERS = 3;
	vector<Layer> lr(N_LAYERS);
	lr[0].init(8, FLOW_SIZE);
	lr[1].init(4, 8);
	lr[2].init(1, 4);

	//открытие окна
	sf::RenderWindow window(sf::VideoMode(1040, 640), "Neural Network", sf::Style::Titlebar | sf::Style::Close);
	sf::Color grey(200, 200, 200);
	sf::Color darkgrey(100, 100, 100);
	sf::Color orange(249, 187, 116);
	sf::Color blue(100, 170, 214);

	//обучение
	bool flag = 1;
	int eph = 0;
	while (flag)
	{
		eph++;
		random_shuffle(pool.begin(), pool.end());

		long double median = 0, cost;
		int correct = 0, accurate = 0;

		//изображение поля
		window.clear();

		//рамка
		sf::RectangleShape shape(sf::Vector2f(501.f, 501.f));
		shape.setOutlineThickness(1);
		shape.setOutlineColor(grey);
		shape.setFillColor(sf::Color::Black);
		shape.setPosition(10.f, 10.f);
		window.draw(shape);

		//оси
		sf::RectangleShape linex(sf::Vector2f(501.f, 1.f));
		linex.setPosition(10.f, 261.f);
		linex.setFillColor(darkgrey);
		window.draw(linex);

		sf::RectangleShape liney(sf::Vector2f(1.f, 501.f));
		liney.setPosition(261.f, 10.f);
		liney.setFillColor(darkgrey);
		window.draw(liney);

		//перебор тестов из выборки
		for (int t = 0; t < BATCH_SIZE; t++)
		{
			//вычисление сети
			lr[0].calculate(pool[t].x);
			for (int i = 1; i < N_LAYERS; i++)
				lr[i].calculate(lr[i - 1].a);
			long double answer = lr[N_LAYERS - 1].a[0];

			//нахождение стоимости и проверка ответа
			cost = pow(pool[t].y - answer, 2);
			if (cost <= ACCURACY)
				accurate++;
			if (round(answer) == pool[t].y)
				correct++;
			median += cost;

			//изображение теста
			sf::CircleShape pixel(2.f);
			pixel.setPosition(pool[t].x[0] * 100 + 260, 260 - pool[t].x[1] * 100);
			if (round(answer))
				pixel.setFillColor(orange);
			else
				pixel.setFillColor(blue);
			window.draw(pixel);

			//градиентный спуск
			vector<long double> expect(1);
			expect[0] = pool[t].y;
			long double derivative = 0;
			for (int i = N_LAYERS - 1; i > 0; i--)
			{
				for (int n = 0; n < lr[i].size; n++)
				{
					derivative = -2 * ETA * (expect[n] - lr[i].a[n]) * lr[i].e[n] * pow(lr[i].a[n], 2);
					for (int k = 0; k < lr[i].inputs; k++)
						lr[i].nr[n].w[k] -= derivative * lr[i - 1].a[k];
					lr[i].nr[n].b -= derivative;
				}
				expect.resize(lr[i].inputs);
				for (int n = 0; n < lr[i].inputs; n++)
				{
					long double sum = 0;
					for (int k = 0; k < lr[i].size; k++)
						sum += -2 * ETA * lr[i].nr[k].w[n] * (expect[k] - lr[i].a[k]) * lr[i].e[k] * pow(lr[i].a[k], 2);
					expect[n] = lr[i - 1].a[n] - sum;
				}
			}
			for (int n = 0; n < lr[0].size; n++)
			{
				derivative = -2 * ETA * (expect[n] - lr[0].a[n]) * lr[0].e[n] * pow(lr[0].a[n], 2);
				for (int k = 0; k < lr[0].inputs; k++)
					lr[0].nr[n].w[k] -= derivative * pool[t].x[k];
				lr[0].nr[n].b -= derivative;
			}
		}

		//номер эпохи
		sf::Font font;
		if (!font.loadFromFile("arial.ttf"))
			cout << "Font loading error";
		sf::Text text;
		text.setFont(font);
		text.setCharacterSize(18);
		text.setFillColor(grey);
		text.setPosition(530.f, 10.f);
		text.setString("Epoch: " + to_string(eph));
		window.draw(text);

		//корректные ответы
		text.setPosition(530.f, 40.f);
		text.setString("Correct: " + to_string(correct) + " / " + to_string(BATCH_SIZE) + " (" + to_string(correct * 100 / BATCH_SIZE) + "%)");
		window.draw(text);

		//аккуратные ответы
		text.setPosition(530.f, 70.f);
		text.setString("Accurate: " + to_string(accurate) + " / " + to_string(BATCH_SIZE) + " (" + to_string(accurate * 100 / BATCH_SIZE) + "%)");
		window.draw(text);

		//среднее отклоненине
		text.setPosition(530.f, 100.f);
		text.setString("Average cost: " + to_string(median / BATCH_SIZE));
		window.draw(text);

		//вывод на экран
		window.display();

		//считывание состояния клавиатуры
		sf::Event event;
		if (window.pollEvent(event))
		{
			if (event.type == sf::Event::LostFocus)
			{
				while (window.waitEvent(event))
					if (event.type == sf::Event::GainedFocus)
						break;
			}
			if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::Space)
				{
					while (window.waitEvent(event))
						if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space)
							break;
						else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C)
						{
							flag = 0;
							break;
						}
				}
				if (event.key.code == sf::Keyboard::C)
					flag = 0;
			}
		}
	}


	//изображение поля
	window.clear(sf::Color::Black);
	window.display();

	//рамка первого поля
	sf::RectangleShape shape(sf::Vector2f(501.f, 501.f));
	shape.setOutlineThickness(1);
	shape.setOutlineColor(grey);
	shape.setFillColor(grey);
	shape.setPosition(10.f, 10.f);
	window.draw(shape);

	//рамка второго поля
	shape.setPosition(530.f, 10.f);
	window.draw(shape);

	//оси
	sf::RectangleShape linex(sf::Vector2f(501.f, 1.f));
	linex.setPosition(10.f, 261.f);
	linex.setFillColor(darkgrey);
	window.draw(linex);

	sf::RectangleShape linex1(sf::Vector2f(501.f, 1.f));
	linex1.setFillColor(darkgrey);
	linex1.setPosition(530.f, 261.f);
	window.draw(linex1);

	sf::RectangleShape liney(sf::Vector2f(1.f, 501.f));
	liney.setPosition(261.f, 10.f);
	liney.setFillColor(darkgrey);
	window.draw(liney);

	sf::RectangleShape liney1(sf::Vector2f(1.f, 501.f));
	liney1.setFillColor(darkgrey);
	liney1.setPosition(781.f, 10.f);
	window.draw(liney1);

	//параметры для статистики
	long double median = 0, cost;
	int correct = 0, accurate = 0;

	//проход по точкам полей
	for (int x = -250; x <= 250; x++)
	{
		for (int y = -250; y <= 250; y++)
		{
			//создание входных данных для заданной точки
			vector<long double> in(FLOW_SIZE);
			in[0] = (long double)x / 100;
			in[1] = (long double)y / 100;
			in[2] = in[0] * in[0];
			in[3] = in[1] * in[1];
			in[4] = sin(in[0] * 100);
			in[5] = sin(in[1] * 100);
			in[6] = in[0] * in[1];

			//вычисление сети
			lr[0].calculate(in);
			for (int i = 1; i < N_LAYERS; i++)
				lr[i].calculate(lr[i - 1].a);
			long double answer = lr[N_LAYERS - 1].a[0];

			//вычисление правильного ответа
			long double right = ((in[1] > 1 / in[0] && in[0] > 0) || (in[1] < 1 / in[0] && in[0] < 0)) ? 1 : 0;

			//нахождение стоимости и проверка ответа
			cost = pow(right - answer, 2);
			if (cost <= ACCURACY)
				accurate++;
			if (round(answer) == right)
				correct++;
			median += cost;

			//закрашивание точки первого поля соответсвенно ответу НС
			sf::CircleShape pixel(1.f);
			if (round(answer))
				pixel.setFillColor(orange);
			else
				pixel.setFillColor(blue);
			pixel.setPosition(260 + x, 260 - y);
			window.draw(pixel);

			//закрашивание точки второго поля соответсвенно правильному ответу
			if (right)
				pixel.setFillColor(orange);
			else
				pixel.setFillColor(blue);
			pixel.setPosition(780 + x, 260 - y);
			window.draw(pixel);

		}
		//вывод осей
		window.draw(linex);
		window.draw(liney);
		window.draw(linex1);
		window.draw(liney1);

		//вывод на экран
		window.display();
	}

	//вывод статистики
	//номер эпохи
	sf::Font font;
	if (!font.loadFromFile("arial.ttf"))
		cout << "Font loading error";
	sf::Text text;
	text.setFont(font);
	text.setCharacterSize(18);
	text.setFillColor(grey);
	text.setPosition(10.f, 520.f);
	text.setString("Completed epochs: " + to_string(eph));
	window.draw(text);

	//корректные ответы
	text.setPosition(10.f, 550.f);
	text.setString("Correct: " + to_string(correct) + " / " + to_string(251001) + " (" + to_string(correct * 100 / 251001) + "%)");
	window.draw(text);

	//аккуратные ответы
	text.setPosition(10.f, 580.f);
	text.setString("Accurate: " + to_string(accurate) + " / " + to_string(251001) + " (" + to_string(accurate * 100 / 251001) + "%)");
	window.draw(text);

	//среднее отклоненине
	text.setPosition(10.f, 610.f);
	text.setString("Average cost: " + to_string(median / 251001));
	window.draw(text);

	//вывод на экран
	window.display();

	//ожидание нажатия на клавиатуру
	sf::Event event;
	while (window.waitEvent(event))
		if (event.type == sf::Event::KeyPressed)
			break;

	//закрытие окна и выход из программы
	window.close();
	return 0;
}