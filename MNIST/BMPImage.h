#ifndef BMPIMAGE_H
#define BMPIMAGE_H

#include <string>
#include <stdexcept>

#ifndef NDEBUG
#include <typeinfo>
#include <cstring>

#endif

/** ------------------数据类型重定义------------------------ */
typedef int int4;              // 4 bytes
typedef unsigned int uint;     // 4 bytes
typedef unsigned char uchar;   // 1 byte
typedef unsigned int uint4;   // 4 bytes
typedef unsigned short uint2;  // 2 bytes
/** -------------------------------------------------------- */

/** ------------------颜色结构体重定义---------------------- */
// Color Palette的结构体，也是RGBA颜色结构体
typedef struct {
    uchar b;
    uchar g;
    uchar r;
    uchar a;
}BGRA,ColorPalette;

// RGB颜色结构体
typedef struct {
    uchar b;
    uchar g;
    uchar r;
}BGR;
/** -------------------------------------------------------- */

class BMPImage {
private:
/** -------------BMP文件头和BMP信息头数据结构定义----------- */
//参考链接如下
    //https://blog.csdn.net/u012877472/article/details/50272771
    //https://blog.csdn.net/lanbing510/article/details/8176231
    //bmp文件头（bmp file header）：共14字节；
    typedef struct {
        uint2 bfType;           //2字节，文件类型；总应是0x4d42('BM')
        uint4 bfSize;           //4字节，文件大小；字节为单位。
        uint2 bfReserved1;      //2字节，保留，必须设置为0
        uint2 bfReserved2;      //2字节，保留，必须设置为0
        uint4 bfOffBits;        //4字节的偏移，表示从文件头到位图数据的偏移
    } BMPFileHeader;

    //位图信息头（bitmap information）：共40字节；
    typedef struct {
        uint4 biSize;           //4字节，信息头的大小，即40；
        int4 biWidth;           //4字节，以像素为单位说明图像的宽度；
        int4 biHeight;          //4字节，以像素为单位说明图像的高度；如果为正，说明位图倒立
        uint2 biPlanes;         //2字节，为目标设备说明颜色平面数，总被设置为1；
        uint2 biBitCount;       //2字节，说明比特数/像素数，值有1、2、4、8、16、24、32；
        uint4 biCompression;    //4字节，说明图像的压缩类型，0(BI_RGB)表示不压缩；
        uint4 biSizeImage;      //4字节，说明位图数据的大小.
        int4 biXPelsPerMeter;   //4字节，表示水平分辨率，单位是像素/米；
        int4 biYPelsPerMeter;   //4字节，表示垂直分辨率，单位是像素/米；
        uint4 biClrUsed;        //4字节，位图使用的调色板中的颜色索引数，为0说明使用所有；
        uint4 biClrImportant;   //4字节，对图像显示有重要影响的颜色索引数，为0说明都重要；
    } BMPInfoHeader;
/** -------------------------------------------------------- */

/** ---------------------公有函数区------------------------- */
public:
    /**
     * 构造函数，初始化一个空的BMPImage
     */
    explicit BMPImage();
    /**
     * 构造函数，载入一张BMP图像,打开失败时会抛出异常
     * @param bmp_file_path 图像路径
     */
    explicit BMPImage(const char* bmp_file_path);
    /**
     * 构造函数，创建大小为width*height的24位的黑色图像，
     * 创建失败时会抛出异常。
     * @param width 宽度
     * @param height 高度
     * @param depth 深度，目前仅支持24位和32位图像
     */
    explicit BMPImage(int4 width, int4 height, uint2 depth = 24);
    /**
     * 拷贝构造函数
     * @param image 图像
     */
    BMPImage(const BMPImage &image);
    /**
     * 赋值构造函数
     * @param image 等号右边的图
     * @return 返回复制的图
     */
    BMPImage &operator=(const BMPImage &image);
    /**
     * 析构函数
     */
    virtual ~BMPImage();
    /**
      * 创建大小为width*height的24位的黑色图像
      * @param width 宽度
      * @param height 高度
      * @param depth 深度，目前仅支持24位图像
      * @return 成功返回true，否则false
      */
    bool create(int4 width,int4 height,uint2 depth=24);
    /**
     * 载入一张BMP图像
     * @param bmp_file_path 图像路径
     * @return 成功返回true，否则false
     */
    bool open(const char* bmp_file_path);
    /**
     * 保存BMP图像
     * @param save_path 保存路径
     * @return 成功返回true，否则false
     */
    bool save(const char* save_path);
    /**
     * 返回(row,col)处像素值的引用
     * @tparam T 可取的值分别为uchar(8位，单通道图像)、
     * BGR(24位，3通道图像)、BGRA(32位，四通道图像)
     * @param row 行
     * @param col 列
     * @return T类型的引用
     */
    template <typename T>
    T & at(uint row, uint col);
    /**
     * 清空图像
     * @return 成功返回true，否则false
     */
    bool clear();

    /**
     * 返回图像宽度
     * @return 宽度
     */
    int4 width();
    /**
     * 返回图像高度
     * @return  高度
     */
    int4 height();
    /**
     * 返回图像深度
     * @return 深度
     */
    uint2 depth();
/** -------------------------------------------------------- */

/** ----------------------私有变量区------------------------ */
private:
    int4 m_width;       //图像宽度
    int4 m_height;      //图像深度
    uint2 m_depth;      //图像深度
    uint4 m_row_size;   //每行字节数
    uchar *m_data;      //图像数据存储区

    BMPFileHeader m_file_header;    //BMP图像的文件头
    BMPInfoHeader m_info_header;    //BMP图像的信息头
    uint m_color_palette_size;      //BMP的调色板数组大小
    ColorPalette *m_color_palette;  //调色板(Color Palette),可选；
/** -------------------------------------------------------- */

/** ------------------------私有函数区------------------------ */
    /**
     * 检查是否越界,以及深度是否匹配
     * @param row 行
     * @param col 列
     * @param depth 深度
     * @return 越界打印错误信息并返回false
     */
     bool valid(int4 row,int4 col,uint2 depth);
/** -------------------------------------------------------- */
};

// 模板函数的实现要放在头文件里
template<typename T>
T &BMPImage::at(uint row, uint col) {
    size_t elem_size = sizeof(T);
#ifdef DEBUG //定义DEBUG时检查
    if(!valid(row,col, (uint2)(elem_size*8))){
        throw std::runtime_error("Index is overflow or T is not match depth");
    }
#endif
    T *value = nullptr;
    // 计算索引
    size_t index = 0;
    if (m_height > 0){
        // 图像数据的存储顺序是从左下到右上，图片上的第一行存在m_data的最后一部分
        index = (m_height-row-1) * m_row_size + col*elem_size;
    }else{
        // 图像数据的存储顺序是从左下到右上，图片上的第一行存在m_data的第一部分
        index = row * m_row_size + col * elem_size;
    }
    value = (T*)(m_data+index);
    return *value;
}

#include <fstream>
#include <iostream>
#include <cstring>

using namespace std;

/**
 * condition为false时抛出错误，错误信息为error_message
 */
#define ASSERT(condition,error_message) ((condition)?0:(assertion(__FILE__, __func__, __LINE__,error_message),-1))

inline void assertion(const std::string &filePath, const std::string &function,
                      int line, const std::string &info){
    //获取文件名
    unsigned long long int pos=filePath.find_last_of('/');
    std::string filename(filePath.substr(pos+1));
    std::string err = filename+" "+function+" "+std::to_string(line)+">>  "+info;
    //抛出错误
    throw std::runtime_error(err);
}

BMPImage::BMPImage():m_width(0),m_height(0),m_depth(8),m_row_size(0),
                     m_data(nullptr),m_color_palette_size(0),m_color_palette(nullptr) {
    // 检查字节数
    ASSERT(sizeof(int4)==4,"The size of 'long' is not 4 bytes");
    ASSERT(sizeof(uint)==4,"The size of 'unsigned int' is not 4 bytes");
    ASSERT(sizeof(uchar)==1,"The size of 'unsigned char' is not 1 bytes");
    ASSERT(sizeof(uint4)==4,"The size of 'unsigned long' is not 4 bytes");
    ASSERT(sizeof(uint2)==2,"The size of 'unsigned short' is not 2 bytes");
    ASSERT(sizeof(BGR)==3,"The size of 'BGR' is not 3 bytes");
    ASSERT(sizeof(BGRA)==4,"The size of 'BGRA' is not 4 bytes");
    ASSERT(sizeof(ColorPalette)==4,"The size of 'ColorPalette' is not 4 bytes");
}

BMPImage::BMPImage(const char* bmp_file_path):BMPImage() {
    char buffer[500];
    sprintf(buffer,"Failed to open bmp image %s\n",bmp_file_path);
    ASSERT(open(bmp_file_path),buffer);
}

BMPImage::BMPImage(int4 width, int4 height, uint2 depth):BMPImage() {
    ASSERT(create(width , height, depth),"Failed to open bmp image\n");
}

BMPImage::BMPImage(const BMPImage &image) {
    // --------------------复制基本属性-------------------
    this->m_width = image.m_width;
    this->m_height = image.m_height;
    this->m_depth = image.m_depth;
    this->m_row_size = image.m_row_size;

    // --------------------复制BMP文件头------------------
    this->m_file_header = image.m_file_header;
    this->m_info_header = image.m_info_header;

    // --------------------复制BMP调色板------------------
    this->m_color_palette_size = image.m_color_palette_size;
    if(m_color_palette_size > 0){
        this->m_color_palette = new ColorPalette[m_color_palette_size];
        memcpy(m_color_palette,image.m_color_palette,m_color_palette_size * sizeof(ColorPalette));
    } else{
        this->m_color_palette = nullptr;
    }

    // --------------------复制图像数据--------------------
    if(m_width>0){
        this->m_data = new uchar[m_info_header.biSizeImage];
        memcpy(this->m_data,image.m_data,m_info_header.biSizeImage);
    } else{
        this->m_data = nullptr;
    }

}

BMPImage &BMPImage::operator=(const BMPImage &image) {
    if(this == &image){
        return *this;
    }
    // --------------------清空原有数据-------------------
    delete[] m_data;
    m_data = nullptr;
    delete[] m_color_palette;
    m_color_palette = nullptr;

    // --------------------复制基本属性-------------------
    this->m_width = image.m_width;
    this->m_height = image.m_height;
    this->m_depth = image.m_depth;
    this->m_row_size = image.m_row_size;

    // --------------------复制BMP文件头------------------
    this->m_file_header = image.m_file_header;
    this->m_info_header = image.m_info_header;

    // --------------------复制BMP调色板------------------
    this->m_color_palette_size = image.m_color_palette_size;
    if(m_color_palette_size>0){
        this->m_color_palette = new ColorPalette[m_color_palette_size];
        memcpy(m_color_palette,image.m_color_palette,m_color_palette_size * sizeof(ColorPalette));
    } else{
        this->m_color_palette = nullptr;
    }

    // --------------------复制图像数据--------------------
    if(m_width>0){
        this->m_data = new uchar[m_info_header.biSizeImage];
        memcpy(this->m_data,image.m_data,m_info_header.biSizeImage);
    } else{
        this->m_data = nullptr;
    }

    return *this;
}

BMPImage::~BMPImage() {
    delete[] m_data;
    delete[] m_color_palette;
}

bool BMPImage::open(const char* bmp_file_path) {
    clear();
    ifstream im_file(bmp_file_path,ios::binary);
    // 检查文件是否打开
    if(!im_file.is_open()){
        printf("Failed to open file %s\n",bmp_file_path);
        return false;
    }
    // -------------------读取BMPFileHeader-------------------------
    char file_header[14];//缓冲区
    im_file.read(file_header, 14);
    // 复制内存
    memcpy(&(m_file_header.bfType),file_header,2);
    memcpy(&(m_file_header.bfSize),file_header+2,4);
    memcpy(&(m_file_header.bfReserved1),file_header+6,2);
    memcpy(&(m_file_header.bfReserved2),file_header+8,2);
    memcpy(&(m_file_header.bfOffBits),file_header+10,4);

    // 判断是否是bmp图像
    if (m_file_header.bfType != 0x4d42) // 0x4d42 = 'BM'
    {
        printf("File %s is not bmp file\n",bmp_file_path);
        return false;
    }
    // --------------------读取BMPInfoHeader------------------------
    char info_header[40];//40字节的BMP信息头缓冲区
    im_file.read(info_header, 40);
    // 复制内存
    memcpy(&(m_info_header.biSize),info_header,4);
    memcpy(&(m_info_header.biWidth),info_header+4,4);
    memcpy(&(m_info_header.biHeight),info_header+8,4);
    memcpy(&(m_info_header.biPlanes),info_header+12,2);
    memcpy(&(m_info_header.biBitCount),info_header+14,2);
    memcpy(&(m_info_header.biCompression),info_header+16,4);
    memcpy(&(m_info_header.biSizeImage),info_header+20,4);
    memcpy(&(m_info_header.biXPelsPerMeter),info_header+24,4);
    memcpy(&(m_info_header.biYPelsPerMeter),info_header+28,4);
    memcpy(&(m_info_header.biClrUsed),info_header+32,4);
    memcpy(&(m_info_header.biClrImportant),info_header+36,4);
    // ---------判断是否有调色板,如果有，则读入调色板数据-----------
    if (m_file_header.bfOffBits == 54){//说明没有调色板
        m_color_palette = nullptr;
    } else{
        // 计算调色板数量
        m_color_palette_size = m_info_header.biClrUsed;
//        m_color_palette_size = (m_file_header.bfOffBits - 54)/ sizeof(ColorPalette);
        m_color_palette = new ColorPalette[m_color_palette_size];
        // 读取调色板数据
        im_file.read((char*)(m_color_palette), m_color_palette_size * sizeof(ColorPalette));

    }
    // 给属性赋值
    m_width = m_info_header.biWidth;
    m_height = m_info_header.biHeight;
    m_depth = m_info_header.biBitCount;
    // 计算每行的字节数
    m_row_size = 4u * ((m_info_header.biBitCount * m_info_header.biWidth+31u)/32u);
    // ---------------------------读取图像数据--------------------------------
    // 申请内存空间并同时初始化为0
    m_data = new uchar[m_info_header.biSizeImage]();
    // 读取图像数据到内存
    im_file.read((char *)m_data,m_info_header.biSizeImage);
    // 关闭文件
    im_file.close();
    return true;
}

bool BMPImage::save(const char* save_path) {
    if (strlen(save_path) == 0){
        printf("Your file path is empty\n");
        return false;
    }
    if (m_width ==0 || m_height == 0){
        printf("This image is empty\n");
        return false;
    }
    ofstream out_image(save_path,ios::binary);
    if (!out_image.is_open()){
        printf("Failed to save image %s\n",save_path);
        return false;
    }
    // ---------------------保存BMP文件头-----------------------
    char file_header[14];
    // 赋值
    memcpy(file_header,&(m_file_header.bfType),2);
    memcpy(file_header+2,&(m_file_header.bfSize),4);
    memcpy(file_header+6,&(m_file_header.bfReserved1),2);
    memcpy(file_header+8,&(m_file_header.bfReserved2),2);
    memcpy(file_header+10,&(m_file_header.bfOffBits),4);
    out_image.write(file_header, 14 * sizeof(char));
    // ---------------------保存BMP信息头-----------------------
    char info_header[40];
    memcpy(info_header,&(m_info_header.biSize),4);
    memcpy(info_header+4,&(m_info_header.biWidth),4);
    memcpy(info_header+8,&(m_info_header.biHeight),4);
    memcpy(info_header+12,&(m_info_header.biPlanes),2);
    memcpy(info_header+14,&(m_info_header.biBitCount),2);
    memcpy(info_header+16,&(m_info_header.biCompression),4);
    memcpy(info_header+20,&(m_info_header.biSizeImage),4);
    memcpy(info_header+24,&(m_info_header.biXPelsPerMeter),4);
    memcpy(info_header+28,&(m_info_header.biYPelsPerMeter),4);
    memcpy(info_header+32,&(m_info_header.biClrUsed),4);
    memcpy(info_header+36,&(m_info_header.biClrImportant),4);
    // 写入
    out_image.write(info_header, 40);
    // ---------------------保存调色板--------------------------
    if(m_color_palette_size > 0){
        out_image.write((char*)m_color_palette,m_color_palette_size* sizeof(ColorPalette));
    }
    // ---------------------保存图像数据------------------------
    out_image.write((char*)m_data, m_info_header.biSizeImage);
    out_image.close();
    return true;
}

bool BMPImage::create(int4 width, int4 height, uint2 depth) {
    clear();
    uint4 bytes_per_pixel = 0;
    switch (depth){
        case 8:
            bytes_per_pixel = 1;
            break;
        case 24:
            bytes_per_pixel = 3;
            break;
        case 32:
            bytes_per_pixel = 4;
            break;
        default:
            ASSERT(false,"The depth must be 8, 24 or 32");
    }

    m_width = width;
    m_height = height;
    m_depth = depth;
    // 计算每行的字节数，每行的字节数需要是4的倍数
    m_row_size = 4u * ((bytes_per_pixel * m_width+4u-1u)/4u);
    // ---------------------初始化BMP文件头---------------------
    m_file_header.bfType = 0x4d42;
    if(depth==8) {
        m_file_header.bfOffBits = 14 + 40 + 256* sizeof(ColorPalette);
    } else {
        m_file_header.bfOffBits = 14 + 40;
    }
    m_file_header.bfSize = m_file_header.bfOffBits + m_height * m_row_size;
    m_file_header.bfReserved1 = 0;
    m_file_header.bfReserved2 = 0;
    // ---------------------初始化BMP信息头----------------------
    m_info_header.biSize = 40;//Should be 40
    m_info_header.biWidth = m_width;
    m_info_header.biHeight = m_height;
    m_info_header.biPlanes = 1;
    m_info_header.biBitCount = m_depth;
    m_info_header.biCompression = 0;
    m_info_header.biSizeImage = m_height * m_row_size;
    m_info_header.biXPelsPerMeter = 3780;
    m_info_header.biYPelsPerMeter = 3780;
    m_info_header.biClrUsed = 0;
    m_info_header.biClrImportant = 0;
    // ------------------调色板------------------------
    if (depth==8){ //8位图像需要调色板
        m_color_palette_size = 256;
        m_color_palette = new ColorPalette[m_color_palette_size];
        for(uint ii=0;ii<m_color_palette_size;ii++){
            m_color_palette[ii].b = (uchar)ii;
            m_color_palette[ii].g = (uchar)ii;
            m_color_palette[ii].r = (uchar)ii;
            m_color_palette[ii].a = 0;
        }
    } else{
        m_color_palette_size = 0;
        m_color_palette = nullptr;
    }
    // ------------------申请内存并初始化为0----------------------
    m_data = new uchar[m_info_header.biSizeImage]();
    return true;
}

bool BMPImage::valid(int4 row, int4 col, uint2 depth) {
    bool valid = true;
    if(row >= (uint)abs(m_height)){
        printf("Index row %d is big than picture height %d-1\n",row,(int)m_height);
        valid = false;
    }
    if(col >= (uint)m_width){
        printf("Index col %d is big than picture width %d-1\n",col,(int)m_width);
        valid = false;
    }
    if(m_depth != depth){
        printf("Error: Not a %dbit image\n", depth);
        valid = false;
    }
    if(!valid)
        clear();
    return valid;
}

bool BMPImage::clear() {
    m_width = 0;
    m_height = 0;
    m_depth = 0;
    delete [] m_data;
    m_data = nullptr;
    delete [] m_color_palette;
    m_color_palette = nullptr;
    return true;
}

int4 BMPImage::width() {
    return abs(m_width);
}

int4 BMPImage::height() {
    return abs(m_height);
}

uint2 BMPImage::depth() {
    return m_depth;
}

#endif //BMPIMAGE_H
