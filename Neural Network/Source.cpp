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

template<typename T>
constexpr auto clause(T n) { return n[1]>0; }
const long double ETA = 0.075, ACCURACY = 0.001;
const int POOL_SIZE = 25000, BATCH_SIZE = 1000, FLOW_SIZE = 7;

long double sigmoid(long double arg)
{
	return 1 / (1 + exp(-arg));
}

class Neuron
{
public:
	long double b, nabla_b = 0;
	vector<long double> w, nabla_w;
	void init(int flows)
	{
		w.resize(flows);
		nabla_w.resize(flows, 0);
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
	int y;
	void init(int flows)
	{
		size = flows;
		x.resize(size);
		for (int i = 0; i < 2; i++)
			x[i] = (long double)(rand() % 495 - 247) / 100;
		x[2] = x[0] * x[0];
		x[3] = x[1] * x[1];
		x[4] = sin(x[0]);
		x[5] = sin(x[1]);
		x[6] = x[0] * x[1];
		y = clause(x);
	}
};

int main()
{
	srand(time(0));
	vector<Input> pool(POOL_SIZE);
	for (int i = 0; i < POOL_SIZE; i++)
		pool[i].init(FLOW_SIZE);

	const int N_LAYERS = 5;
	vector<Layer> lr(N_LAYERS);
	lr[0].init(8, FLOW_SIZE);
	lr[1].init(8, 8);
	lr[2].init(8, 8);
	lr[3].init(8, 8);
	//lr[4].init(4, 4);
	//lr[5].init(4, 4);
	//lr[6].init(4, 4);
	//lr[7].init(4, 4);
	//lr[8].init(4, 4);
	lr[N_LAYERS-1].init(1, 8);

	//открытие окна
	sf::RenderWindow window(sf::VideoMode(1200, 600), "Neural Network", sf::Style::Titlebar | sf::Style::Close);
	sf::Color grey(200, 200, 200);
	sf::Color darkgrey(100, 100, 100);
	sf::Color orange(249, 187, 116);
	sf::Color blue(100, 170, 214);

	//*********************
	bool flag = 1;
	int eph = 0;
	while (flag)
	{
		eph++;
		random_shuffle(pool.begin(), pool.end());

		long double median = 0, cost;
		int correct = 0, accurate = 0;

		//рисование поля
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

			//рисование теста
			sf::CircleShape pixel(2.f);
			pixel.setPosition(pool[t].x[0] * 100 + 260, 260 - pool[t].x[1] * 100);
			pixel.setFillColor(sf::Color(100 + answer * 149, 170 + answer * 17, 214 - answer * 98));
			window.draw(pixel);

			//обучение
			vector<long double> expect(1);
			expect[0] = pool[t].y;
			long double derivative = 0;
			for (int i = N_LAYERS - 1; i > 0; i--)
			{
				for (int n = 0; n < lr[i].size; n++)
				{
					derivative = -2 * ETA * (expect[n] - lr[i].a[n]) * lr[i].e[n] * pow(lr[i].a[n], 2);
					for (int k = 0; k < lr[i].inputs; k++)
						lr[i].nr[n].nabla_w[k] -= derivative * lr[i - 1].a[k];
					lr[i].nr[n].nabla_b -= derivative;
				}
				vector<long double> nexpect(lr[i].inputs);
				for (int n = 0; n < lr[i].inputs; n++)
				{
					long double sum = 0;
					for (int k = 0; k < lr[i].size; k++)
					{
						long double htu = (expect[k] - lr[i].a[k]) * lr[i].e[k] * pow(lr[i].a[k], 2);
						sum += -2 * ETA * lr[i].nr[k].w[n] * htu;
						if (isnan(sum))
						{
							cout << "Shit";
							if (isnan((expect[k] - lr[i].a[k]) * lr[i].e[k] * pow(lr[i].a[k], 2)))
								cout << "1";
							if (isnan(lr[i].nr[k].w[n] * (expect[k] - lr[i].a[k]) * lr[i].e[k] * pow(lr[i].a[k], 2)))
								cout << "2";
							long double htu = (expect[k] - lr[i].a[k]) * lr[i].e[k] * pow(lr[i].a[k], 2);
							if (isnan(lr[i].nr[k].w[n] * htu))
								cout << "3";
							return 0;
						}
					}
					nexpect[n] = lr[i - 1].a[n] - sum;
				}
				expect = nexpect;
			}
			for (int n = 0; n < lr[0].size; n++)
			{
				derivative = -2 * ETA * (expect[n] - lr[0].a[n]) * lr[0].e[n] * pow(lr[0].a[n], 2);
				for (int k = 0; k < lr[0].inputs; k++)
					lr[0].nr[n].nabla_w[k] -= derivative * pool[t].x[k];
				lr[0].nr[n].nabla_b -= derivative;
			}
		}
		
		//номер эпохи
		sf::Font font;
		if (!font.loadFromFile("arial.ttf"))
			cout << "Failed to load font";
		sf::Text text;
		text.setFont(font);
		text.setCharacterSize(18);
		text.setFillColor(grey);
		text.setPosition(530.f, 10.f);
		text.setString("Epoch:    " + to_string(eph));
		window.draw(text);

		//Корректные ответы
		text.setPosition(530.f, 40.f);
		text.setString("Correct:  " + to_string(correct) + "/" + to_string(BATCH_SIZE) + " (" + to_string(correct * 100 / BATCH_SIZE) + "%)");
		window.draw(text);

		//Аккуратные ответы
		text.setPosition(530.f, 70.f);
		text.setString("Accurate: " + to_string(accurate) + "/" + to_string(BATCH_SIZE) + " (" + to_string(accurate * 100 / BATCH_SIZE) + "%)");
		window.draw(text);

		//среднее отклоненине
		text.setPosition(530.f, 100.f);
		text.setString("Average cost: " + to_string(median / BATCH_SIZE));
		window.draw(text);

		//вывод на экран
		window.display();

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

		//обучене
		for (int i = N_LAYERS - 1; i >= 0; i--)
		{
			for (int n = 0; n < lr[i].size; n++)
			{
				for (int k = 0; k < lr[i].inputs; k++)
					lr[i].nr[n].w[k] += lr[i].nr[n].nabla_w[k]/BATCH_SIZE;
				lr[i].nr[n].nabla_w.resize(lr[i].inputs, 0);
				lr[i].nr[n].b += lr[i].nr[n].nabla_b/BATCH_SIZE;
				lr[i].nr[n].nabla_b = 0;
			}
		}

	}

	//рисование поля
	window.clear(sf::Color::Black);
	window.display();

	//рамка
	sf::RectangleShape shape(sf::Vector2f(501.f, 501.f));
	shape.setOutlineThickness(1);
	shape.setOutlineColor(grey);
	shape.setFillColor(grey);
	shape.setPosition(10.f, 10.f);
	window.draw(shape);

	shape.setPosition(530.f, 10.f);
	window.draw(shape);

	//оси
	sf::RectangleShape linex(sf::Vector2f(501.f, 1.f));
	linex.setPosition(10.f, 261.f);
	linex.setFillColor(darkgrey);
	window.draw(linex);

	sf::RectangleShape linex1(sf::Vector2f(501.f, 1.f));
	linex1.setPosition(530.f, 261.f);
	linex1.setFillColor(darkgrey);
	window.draw(linex1);

	sf::RectangleShape liney(sf::Vector2f(1.f, 501.f));
	liney.setPosition(261.f, 10.f);
	liney.setFillColor(darkgrey);
	window.draw(liney);

	sf::RectangleShape liney1(sf::Vector2f(1.f, 501.f));
	liney1.setFillColor(darkgrey);
	liney1.setPosition(781.f, 10.f);
	window.draw(liney1);

	//поле
	for (int x = -250; x <= 250; x++)
	{
		for (int y = -250; y <= 250; y++)
		{
			vector<long double> in(7);
			in[0] = (float)x / 100;
			in[1] = (float)y / 100;
			in[2] = in[0] * in[0];
			in[3] = in[1] * in[1];
			in[4] = sin(in[0]);
			in[5] = sin(in[1]);
			in[6] = in[0] * in[1];

			lr[0].calculate(in);
			for (int i = 1; i < N_LAYERS; i++)
				lr[i].calculate(lr[i - 1].a);
			long double answer = lr[N_LAYERS - 1].a[0];

			//график по данным нейросети
			sf::CircleShape pixel(1.f);
			pixel.setFillColor(sf::Color(100 + answer * 149, 170 + answer * 17, 214 - answer * 98));
			pixel.setPosition(260 + x, 260 - y);
			window.draw(pixel);

			//правильный график
			answer = clause(in);
			if (round(answer))
				pixel.setFillColor(orange);
			else
				pixel.setFillColor(blue);
			pixel.setPosition(780 + x, 260 - y);
			window.draw(pixel);
		}

		window.draw(linex);
		window.draw(liney);

		window.draw(linex1);
		window.draw(liney1);

		window.display();
	}

	//закрытие
	sf::Event event;
	while (window.waitEvent(event))
		if (event.type == sf::Event::KeyPressed)
			break;
	window.close();
	return 0;
}