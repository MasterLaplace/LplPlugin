#include <math.h>
#include <stdio.h>
#include <stdint.h>

float ratio(float x, float y)
{
    float result = x / y;
    if (result > 1.0f)
    {
        printf("x(%.2f) est %.2f fois plus fort que y(%.2f)\n", x, result, y);
        printf("  et: y(%.2f) est %.2f fois plus faible que x(%.2f)\n", y, result, x);
    }
    else if(result == 1.0f)
        printf("x(%.2f) égale à y(%.2f)\n", x, y);
    else
    {
        printf("y(%.2f) est %.2f fois plus fort que x(%.2f)\n", y, y / x, x);
        printf("  et: x(%.2f) est %.2f fois plus fort que y(%.2f)\n", x, result, y);
        printf("  donc: x(%.2f) est %.2f fois plus faible que y(%.2f)\n", x, y / x, y);
    }
    return result;
}

/*
float x = 10.f;
float y = 2.f;
ratio(x, y);
ratio(y, x);
ratio(y, y);
*/

float integrale(uint16_t lower_b, uint16_t upper_b, float *curve)
{
    float area = 0.f;
    for (uint16_t i_axis = lower_b; i_axis <= upper_b; ++i_axis)
        area += curve[i_axis];

    return area;
}

float schumacher(uint8_t nb_channels, float channels[nb_channels][150])
{
    float score = 0.f;
    for (uint8_t i = 0u; i < nb_channels; ++i)
        score += integrale(40u, 70u, channels[i]);
    score *= (1.f / nb_channels);
    return score;
}

void main(void)
{
    const uint8_t nb_channels = 8u;
    float channels[8][150] = {{0}};

    schumacher(nb_channels, channels);
}

// Build: gcc calcul.c -o calcul
