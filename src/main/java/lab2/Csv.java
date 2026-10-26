package lab2;

/**
 * 共享的 CSV 解析辅助方法。
 *
 * <p>train.csv 与 test.csv 均以逗号分隔；train.csv 比 test.csv 多一列 bad_good。
 * 所有缺失值 / 非法数值按 0 处理。
 */
public final class Csv {

    private Csv() {
    }

    /** 7 类细分资产的中文名称（与列索引一一对应）。 */
    public static final String[] ASSET_TYPE_NAMES =
            {"存款", "理财", "私人理财", "基金", "国债", "银保通", "贵金属"};

    /** train.csv 中 7 类细分资产（存款…贵金属）的列索引。 */
    public static final int[] TRAIN_ASSET_TYPE_IDX = {3, 4, 5, 6, 7, 8, 9};

    /** 任务5 使用的 10 个连续特征在 train.csv 中的列索引。 */
    public static final int[] TRAIN_FEAT_IDX = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    /** 任务5 使用的 10 个连续特征在 test.csv 中的列索引。 */
    public static final int[] TEST_FEAT_IDX = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    /** 判断一行是否为表头。 */
    public static boolean isHeader(String[] f) {
        return f.length > 0 && "CUST_ID".equals(f[0].trim());
    }

    public static double toDouble(String s) {
        if (s == null || s.isEmpty()) {
            return 0.0;
        }
        try {
            return Double.parseDouble(s.trim());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }

    public static long toLong(String s) {
        if (s == null || s.isEmpty()) {
            return 0L;
        }
        try {
            return Long.parseLong(s.trim());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    public static double fieldDouble(String[] f, int idx) {
        return idx < f.length ? toDouble(f[idx]) : 0.0;
    }

    public static long fieldLong(String[] f, int idx) {
        return idx < f.length ? toLong(f[idx]) : 0L;
    }

    public static String fieldStr(String[] f, int idx) {
        return idx < f.length ? f[idx].trim() : "";
    }
}
