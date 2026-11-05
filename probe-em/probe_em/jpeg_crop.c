/* Assemble a cropped grayscale precomputed JPEG stack without requantization.
 * All XY boundaries must lie on the original 8x8 DCT grid. The caller supplies
 * nonoverlapping input tiles, their plane heights and offsets in the output.
 * Build: cc -O2 jpeg_crop.c -o jpeg_crop -ljpeg
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <jpeglib.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

static uint32_t read_u32(void) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, stdin) != 4) {
        fprintf(stderr, "Truncated JPEG input stream\n"); exit(2);
    }
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static int dimension(const char *text, int positive) {
    char *end;
    long value = strtol(text, &end, 10);
    if (!*text || *end || value < (positive ? 1 : -65535) || value > 65535) {
        fprintf(stderr, "Invalid JPEG dimension or offset\n"); exit(2);
    }
    return (int)value;
}

int main(int argc, char **argv) {
    int streaming = argc == 5 && !strcmp(argv[1], "--stream");
    if (!streaming && (argc < 9 || (argc - 5) % 4 || argc > 21)) {
        fprintf(stderr, "Usage: jpeg_crop output width height z [input plane_y offset_x offset_y]...\n");
        fprintf(stderr, "Or: jpeg_crop --stream width height z < framed-input > output.jpg\n");
        return 2;
    }
#ifdef _WIN32
    if (streaming) { _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY); }
#endif
    int dx = dimension(argv[2], 1), dy = dimension(argv[3], 1), sz = dimension(argv[4], 1);
    if (dx % 8 || dy % 8 || (long)dy * sz > 65535) {
        fprintf(stderr, "Output must preserve complete grayscale DCT blocks\n"); return 2;
    }
    int count = streaming ? (int)read_u32() : (argc - 5) / 4;
    if (count < 1 || count > 4) { fprintf(stderr, "Expected one to four JPEG tiles\n"); return 2; }
    struct jpeg_decompress_struct source[4];
    struct jpeg_error_mgr errors[4], target_error;
    jvirt_barray_ptr *original[4], cropped[1];
    FILE *input[4];
    unsigned char *buffers[4] = {NULL, NULL, NULL, NULL};
    struct jpeg_compress_struct target;
    target.err = jpeg_std_error(&target_error);
    jpeg_create_compress(&target);
    size_t copied = 0;
    for (int piece = 0; piece < count; ++piece) {
        int sy = streaming ? (int)read_u32() : dimension(argv[6 + 4 * piece], 1);
        int ox = streaming ? (int32_t)read_u32() : dimension(argv[7 + 4 * piece], 0);
        int oy = streaming ? (int32_t)read_u32() : dimension(argv[8 + 4 * piece], 0);
        if (sy < 1 || sy > 65535 || ox < -65535 || ox > 65535 || oy < -65535 || oy > 65535
                || sy % 8 || ox % 8 || oy % 8) {
            fprintf(stderr, "Input tiles must share the output DCT grid\n"); return 2;
        }
        struct jpeg_decompress_struct *s = &source[piece];
        s->err = jpeg_std_error(&errors[piece]);
        jpeg_create_decompress(s);
        if (streaming) {
            uint32_t length = read_u32();
            if (!length || length > 268435456) { fprintf(stderr, "Invalid JPEG payload length\n"); return 2; }
            buffers[piece] = malloc(length);
            if (!buffers[piece] || fread(buffers[piece], 1, length, stdin) != length) {
                fprintf(stderr, "Cannot read JPEG input payload\n"); return 2;
            }
            jpeg_mem_src(s, buffers[piece], length);
        } else {
            input[piece] = fopen(argv[5 + 4 * piece], "rb");
            if (!input[piece]) { perror("JPEG input"); return 2; }
            jpeg_stdio_src(s, input[piece]);
        }
        jpeg_read_header(s, TRUE);
        if (s->num_components != 1 || s->jpeg_color_space != JCS_GRAYSCALE ||
            s->image_height != (unsigned)sy * sz || s->image_width % 8 ||
            s->comp_info[0].h_samp_factor != 1 || s->comp_info[0].v_samp_factor != 1) {
            fprintf(stderr, "Input is not a compatible grayscale XYZ stack\n"); return 2;
        }
        original[piece] = jpeg_read_coefficients(s);
        if (!piece) {
            jpeg_copy_critical_parameters(s, &target);
            cropped[0] = (*s->mem->request_virt_barray)((j_common_ptr)s,
                JPOOL_IMAGE, TRUE, dx / 8, dy * sz / 8, 1);
            (*s->mem->realize_virt_arrays)((j_common_ptr)s);
            /* Initialize all rows before writing tiles that skip rows between
             * planes; libjpeg forbids jumping past the first undefined row. */
            for (int row = 0; row < dy * sz / 8; ++row) {
                JBLOCKARRAY out = (*s->mem->access_virt_barray)((j_common_ptr)s,
                    cropped[0], row, 1, TRUE);
                memset(out[0], 0, (dx / 8) * sizeof(JBLOCK));
            }
        } else {
            JQUANT_TBL *a = source[0].quant_tbl_ptrs[source[0].comp_info[0].quant_tbl_no];
            JQUANT_TBL *b = s->quant_tbl_ptrs[s->comp_info[0].quant_tbl_no];
            if (!a || !b || memcmp(a->quantval, b->quantval, sizeof(a->quantval))) {
                fprintf(stderr, "Input JPEG quantization tables differ\n"); return 2;
            }
        }
        int x0 = ox < 0 ? 0 : ox, y0 = oy < 0 ? 0 : oy;
        int x1 = ox + (int)s->image_width, y1 = oy + sy;
        if (x1 > dx) x1 = dx;
        if (y1 > dy) y1 = dy;
        if (x1 <= x0 || y1 <= y0) { fprintf(stderr, "Empty input intersection\n"); return 2; }
        for (int z = 0; z < sz; ++z) {
            for (int y = y0 / 8; y < y1 / 8; ++y) {
                JBLOCKARRAY in = (*s->mem->access_virt_barray)((j_common_ptr)s,
                    original[piece][0], z * (sy / 8) + y - oy / 8, 1, FALSE);
                JBLOCKARRAY out = (*source[0].mem->access_virt_barray)((j_common_ptr)&source[0],
                    cropped[0], z * (dy / 8) + y, 1, TRUE);
                memcpy(out[0] + x0 / 8, in[0] + (x0 - ox) / 8, ((x1 - x0) / 8) * sizeof(JBLOCK));
            }
        }
        copied += (size_t)(x1 - x0) * (y1 - y0) * sz;
    }
    if (copied != (size_t)dx * dy * sz) {
        fprintf(stderr, "Input tiles do not cover the output exactly\n"); return 2;
    }
    target.image_width = dx;
    target.image_height = dy * sz;
#if JPEG_LIB_VERSION >= 70
    target.jpeg_width = dx;
    target.jpeg_height = dy * sz;
#endif
    target.optimize_coding = FALSE;
    FILE *output = NULL;
    unsigned char *encoded = NULL;
    unsigned long encoded_size = 0;
    if (streaming) {
        jpeg_mem_dest(&target, &encoded, &encoded_size);
    } else {
        output = fopen(argv[1], "wb");
        if (!output) { perror("JPEG output"); return 2; }
        jpeg_stdio_dest(&target, output);
    }
    jpeg_write_coefficients(&target, cropped);
    jpeg_finish_compress(&target);
    jpeg_destroy_compress(&target);
    for (int piece = 0; piece < count; ++piece) {
        jpeg_finish_decompress(&source[piece]);
        jpeg_destroy_decompress(&source[piece]);
        if (streaming) free(buffers[piece]);
        else fclose(input[piece]);
    }
    if (streaming) {
        if (fwrite(encoded, 1, encoded_size, stdout) != encoded_size || fflush(stdout)) {
            fprintf(stderr, "Cannot write JPEG output stream\n"); return 2;
        }
        free(encoded);
    } else if (fclose(output)) { perror("JPEG close"); return 2; }
    return 0;
}
