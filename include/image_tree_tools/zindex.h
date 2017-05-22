#ifndef ZINDEX_H_
#define ZINDEX_H_

#include <cmath>
#include <vector>

#include <stdint.h>

/*!
 *  Computes the morton number for three 10-bit integers
 *
 *  \param x  Integer that uses up to 10 bit
 *  \param y  Integer that uses up to 10 bit
 *  \param z  Integer that uses up to 10 bit
 *
 *  \return   The morton number as 32-bit int with 30 bits used.
 */
inline uint32_t morton_3d(uint32_t x, uint32_t y, uint32_t z)
{
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x <<  8)) & 0x0300F00F;
  x = (x | (x <<  4)) & 0x030C30C3;
  x = (x | (x <<  2)) & 0x09249249;

  y = (y | (y << 16)) & 0x030000FF;
  y = (y | (y <<  8)) & 0x0300F00F;
  y = (y | (y <<  4)) & 0x030C30C3;
  y = (y | (y <<  2)) & 0x09249249;

  z = (z | (z << 16)) & 0x030000FF;
  z = (z | (z <<  8)) & 0x0300F00F;
  z = (z | (z <<  4)) & 0x030C30C3;
  z = (z | (z <<  2)) & 0x09249249;

  return x | (y << 1) | (z << 2);
}


/*!
 *  Computes the non-interleaved inputs from the given morton number
 *
 *  \param x      Output parameter. stores as 10-bit integer
 *  \param y      Output parameter. stores as 10-bit integer
 *  \param z      Output parameter. stores as 10-bit integer
 *  \param input  Input morton number with 30 bits. The two most significant
 *                bits must be 0.
 */
inline void inverse_morton_3d(uint32_t& x, uint32_t& y, uint32_t& z, uint32_t input)
{
  x = input &        0x09249249;
  y = (input >> 1) & 0x09249249;
  z = (input >> 2) & 0x09249249;

  x = ((x >> 2) | x) & 0x030C30C3;
  x = ((x >> 4) | x) & 0x0300F00F;
  x = ((x >> 8) | x) & 0x030000FF;
  x = ((x >>16) | x) & 0x000003FF;

  y = ((y >> 2) | y) & 0x030C30C3;
  y = ((y >> 4) | y) & 0x0300F00F;
  y = ((y >> 8) | y) & 0x030000FF;
  y = ((y >>16) | y) & 0x000003FF;

  z = ((z >> 2) | z) & 0x030C30C3;
  z = ((z >> 4) | z) & 0x0300F00F;
  z = ((z >> 8) | z) & 0x030000FF;
  z = ((z >>16) | z) & 0x000003FF;
}


#endif /* ZINDEX_H_ */
