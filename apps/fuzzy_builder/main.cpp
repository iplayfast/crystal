#ifndef Q_MOC_RUN
#include "crystal/fuzzy/fuzzy_set.hpp"
#endif

#include <QApplication>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QTextEdit>
#include <QGroupBox>
#include <QFormLayout>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QStatusBar>
#include <QToolBar>
#include <QScrollBar>
#include <QDoubleSpinBox>

#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cassert>

// ── Data Model ──────────────────────────────────────────────────────────────

enum class BlockType { In = 0, Out = 1, Or = 2, Xor = 3, And = 4, Fuzzy = 5, Package = 6 };

static const char* block_type_name(BlockType t) {
    switch (t) {
        case BlockType::In:      return "In";
        case BlockType::Out:     return "Out";
        case BlockType::Or:      return "Or";
        case BlockType::Xor:     return "Xor";
        case BlockType::And:     return "And";
        case BlockType::Fuzzy:   return "Fuzzy";
        case BlockType::Package: return "Package";
    }
    return "";
}

static QColor block_color(BlockType t) {
    switch (t) {
        case BlockType::In:      return QColor(100, 180, 100);
        case BlockType::Out:     return QColor(180, 100, 100);
        case BlockType::Or:      return QColor(100, 100, 200);
        case BlockType::Xor:     return QColor(180, 100, 180);
        case BlockType::And:     return QColor(200, 180, 80);
        case BlockType::Fuzzy:   return QColor(80, 180, 200);
        case BlockType::Package: return QColor(160, 160, 160);
    }
    return Qt::gray;
}

struct LogicBlock {
    int id = 0;
    BlockType type = BlockType::In;
    std::string name = "No_Name";
    std::string description;
    std::string function_text = "return 0;\n";
    int x = 0, y = 0;
    int imin = 0, imax = 255, omin = 0, omax = 255;
    float in_value = -1, out_value = -1;
    bool selected = false;
    bool packed = false;
    int pack_id = -1;
    bool done = false;
    std::optional<crystal::FuzzySet<float>> fuzzy;
};

struct Connection {
    int from_id = -1;
    int to_id = -1;
};

// ── LogicGraph ──────────────────────────────────────────────────────────────

class LogicGraph {
public:
    std::vector<LogicBlock> blocks;
    std::vector<Connection> connections;
    std::string global_code = "\n";

    LogicBlock* get(int id) {
        for (auto& b : blocks) if (b.id == id) return &b;
        return nullptr;
    }
    const LogicBlock* get(int id) const {
        for (auto& b : blocks) if (b.id == id) return &b;
        return nullptr;
    }

    int unique_id() const {
        int id = 0;
        bool found;
        do {
            found = false;
            for (auto& b : blocks) {
                if (b.id == id) { id++; found = true; break; }
            }
        } while (found);
        return id;
    }

    int count(BlockType t) const {
        int n = 0;
        for (auto& b : blocks) if (b.type == t) n++;
        return n;
    }

    void unselect_all() {
        for (auto& b : blocks) b.selected = false;
    }

    void remove_block(int id) {
        // Remove connections referencing this block
        connections.erase(
            std::remove_if(connections.begin(), connections.end(),
                [id](const Connection& c) { return c.from_id == id || c.to_id == id; }),
            connections.end());
        blocks.erase(
            std::remove_if(blocks.begin(), blocks.end(),
                [id](const LogicBlock& b) { return b.id == id; }),
            blocks.end());
    }

    void remove_connection(int from_id, int to_id) {
        connections.erase(
            std::remove_if(connections.begin(), connections.end(),
                [from_id, to_id](const Connection& c) {
                    return c.from_id == from_id && c.to_id == to_id;
                }),
            connections.end());
    }

    void clear() {
        blocks.clear();
        connections.clear();
        global_code = "\n";
    }

    // ── Recursive Logic Evaluation (from original) ─────────────────────────

    float evaluate_node(int id, int depth) {
        if (depth > 50) return -1; // infinite recursion guard
        auto* node = get(id);
        if (!node) return -1;

        // Gather inputs to this node
        for (auto& c : connections) {
            if (c.to_id == id) {
                auto* from = get(c.from_id);
                if (!from) continue;
                float val;
                if (from->done) {
                    val = from->out_value;
                } else {
                    val = evaluate_node(c.from_id, depth + 1);
                }
                node->in_value = val;
                // Calculate based on type
                switch (node->type) {
                    case BlockType::In:
                        node->out_value = val;
                        break;
                    case BlockType::Out:
                        node->out_value = val;
                        break;
                    case BlockType::Or:
                        if (node->out_value < 0) node->out_value = val;
                        else node->out_value = std::max(node->out_value, val);
                        break;
                    case BlockType::Xor:
                        if (node->out_value < 0) node->out_value = val;
                        else node->out_value = std::max(node->in_value, node->out_value) -
                                               std::min(node->in_value, node->out_value);
                        break;
                    case BlockType::And:
                        if (node->out_value < 0) node->out_value = val;
                        else node->out_value = std::min(node->out_value, val);
                        break;
                    case BlockType::Fuzzy:
                        if (node->fuzzy)
                            node->out_value = node->fuzzy->evaluate(val);
                        break;
                    case BlockType::Package:
                        break;
                }
            }
        }
        node->done = true;
        return node->out_value;
    }

    void update_logic() {
        for (auto& b : blocks) {
            b.done = false;
            if (b.type != BlockType::In) {
                b.in_value = -1;
                b.out_value = -1;
            } else {
                b.out_value = b.in_value;
            }
        }
        for (auto& b : blocks) {
            if (b.type == BlockType::Out || !has_output_connection(b.id)) {
                b.out_value = evaluate_node(b.id, 0);
                b.done = true;
            }
        }
    }

    bool has_output_connection(int id) const {
        for (auto& c : connections) if (c.from_id == id) return true;
        return false;
    }

    // ── Save / Load (original format) ──────────────────────────────────────

    void save(const std::string& path) const {
        std::ofstream f(path);
        if (!f) return;
        f << "// The next lines are used by the fuzzy editor to save and load data\n";
        f << "/*===FuzzyBuilder Data Start===\n";
        f << blocks.size() << " Number of logic blocks\n";
        for (auto& b : blocks) save_block(f, b);

        f << "\n" << connections.size() << " Connection Count\n";
        for (auto& c : connections) {
            auto* fb = get(c.from_id);
            auto* tb = get(c.to_id);
            if (fb && tb)
                f << c.from_id << " " << c.to_id
                  << " Connection from " << fb->name << " to " << tb->name << "\n";
        }

        // Global code
        f << "\n";
        int line_count = 0;
        for (char ch : global_code) if (ch == '\n') line_count++;
        f << line_count << " Count of lines of Global Code\n";
        std::istringstream ss(global_code);
        std::string line;
        while (std::getline(ss, line)) {
            f << "\\\\" << line << "\n";
        }

        f << "*/\n\n/*The next section is C header code to include with your source code*/\n\n";
        export_c_header(f);
    }

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return;
        std::string line;
        // Find marker
        while (std::getline(f, line)) {
            if (line.find("/*===FuzzyBuilder Data Start===") != std::string::npos) break;
        }
        clear();

        // Block count
        std::getline(f, line);
        int block_count = 0;
        std::sscanf(line.c_str(), "%d", &block_count);

        int max_id = -1;
        for (int i = 0; i < block_count; i++) {
            LogicBlock b;
            load_block(f, b);
            if (b.id > max_id) max_id = b.id;
            blocks.push_back(std::move(b));
        }

        // Connection count
        std::getline(f, line); // blank
        std::getline(f, line);
        int conn_count = 0;
        std::sscanf(line.c_str(), "%d", &conn_count);
        for (int i = 0; i < conn_count; i++) {
            std::getline(f, line);
            if (line.empty() || line == "*/") break;
            int from = -1, to = -1;
            std::sscanf(line.c_str(), "%d %d", &from, &to);
            if (from >= 0 && to >= 0 && from != to && from <= max_id && to <= max_id) {
                if (get(from) && get(to))
                    connections.push_back({from, to});
            }
        }

        // Global code
        std::getline(f, line); // blank
        std::getline(f, line);
        int global_lines = 0;
        std::sscanf(line.c_str(), "%d", &global_lines);
        global_code.clear();
        for (int i = 0; i < global_lines; i++) {
            std::getline(f, line);
            if (line.size() >= 2 && line[0] == '\\' && line[1] == '\\')
                line = line.substr(2);
            global_code += line + "\n";
        }
    }

    // ── C Header Export ────────────────────────────────────────────────────

    void export_c_header(std::ostream& out) const {
        // Find largest fuzzy data length
        int largest = 0;
        for (auto& b : blocks) {
            if (b.type == BlockType::Fuzzy && b.fuzzy) {
                largest = std::max(largest, (int)b.fuzzy->size());
            }
        }
        out << "#ifndef FUZZY_DATA_LENGTH\n#define FUZZY_DATA_LENGTH " << largest << "\n#endif\n";
        out << "#include \"TFuzzy.h\"\n";
        out << "#ifdef __cplusplus\nusing namespace Crystal;\n#endif\n\n";
        out << "/* Global information entered by you */\n" << global_code << "\n";

        // Prototypes
        out << "\n/* Function prototypes */\n";
        for (auto& b : blocks) {
            write_header(out, b, true);
        }

        // Output functions
        out << "\n\n/* Output Functions */\n";
        for (auto& b : blocks) {
            if (b.type == BlockType::Out) {
                write_header(out, b, false);
                for (auto& c : connections) {
                    if (c.to_id == b.id) {
                        auto* from = get(c.from_id);
                        if (from) out << "  return " << from->name << "();\n";
                    }
                }
                out << "\n}\n";
            }
        }

        // Non-output functions
        out << "\n\n/* Non-Output Functions */\n";
        for (auto& b : blocks) {
            switch (b.type) {
                case BlockType::In:
                    write_header(out, b, false);
                    out << "\n" << b.function_text << "\n}\n\n";
                    break;
                case BlockType::Or:
                case BlockType::And:
                case BlockType::Xor: {
                    write_header(out, b, false);
                    bool first = true;
                    for (auto& c : connections) {
                        if (c.to_id == b.id) {
                            auto* from = get(c.from_id);
                            if (!from) continue;
                            if (first) {
                                out << "VALUE_TYPE max,min = " << from->name << "();\n  max = min;\n";
                                first = false;
                            } else {
                                out << "  {\n  VALUE_TYPE in = " << from->name << "();\n"
                                    << "    max = MAXVT(in,max);\n    min = MINVT(in,min);\n  }\n";
                            }
                        }
                    }
                    if (b.type == BlockType::Xor) out << "  return max - min;\n}\n";
                    else if (b.type == BlockType::Or) out << "  return max;\n}\n\n";
                    else out << "  return min;\n}\n\n";
                    break;
                }
                case BlockType::Fuzzy: {
                    write_header(out, b, false);
                    if (b.fuzzy) {
                        out << "static struct TFuzzy f = {" << b.fuzzy->size() << ",\n\t\t{";
                        for (size_t i = 0; i < b.fuzzy->size(); i++) {
                            if (i > 0) out << ",";
                            out << "{" << b.fuzzy->x_at(i) << "," << b.fuzzy->membership_at(i) << "}";
                            if (i % 5 == 0) out << "\n\t\t";
                        }
                        out << "}};\n";
                    }
                    for (auto& c : connections) {
                        if (c.to_id == b.id) {
                            auto* from = get(c.from_id);
                            if (from) {
                                out << "  return Value(&f," << from->name << "());\n}\n\n";
                                break;
                            }
                        }
                    }
                    break;
                }
                default: break;
            }
        }
    }

private:
    static void write_header(std::ostream& out, const LogicBlock& b, bool proto) {
        out << "\n/******************************************************";
        out << "\n* Description of " << b.name << " Type " << block_type_name(b.type);
        out << "\n* " << b.description;
        out << "\n*******************************************************/";
        out << "\nVALUE_TYPE " << b.name << "(void)" << (proto ? ";" : "\n{") << "\n";
    }

    static void save_block(std::ostream& f, const LogicBlock& b) {
        f << "\nID " << b.id << " Type " << (int)b.type << " (" << block_type_name(b.type) << ")\n";
        f << "Packed " << (b.packed ? 1 : 0) << " PackageID " << b.pack_id << "\n";
        f << b.x << " " << b.y << " X,Y location on screen\n";
        if (b.type == BlockType::Fuzzy) {
            f << b.imin << " " << b.imax << " Min and Max of inputs\n";
            f << b.omin << " " << b.omax << " Min and Max of outputs\n";
        } else if (b.type == BlockType::In) {
            f << b.imin << " " << b.imax << " Min and Max of inputs\n";
        }
        f << "Name: " << b.name << "\nDescription: " << b.description << "\n";
        if (b.type == BlockType::In) {
            int line_count = 0;
            for (char ch : b.function_text) if (ch == '\n') line_count++;
            f << line_count << " Count of lines of user code\n";
            std::istringstream ss(b.function_text);
            std::string line;
            while (std::getline(ss, line)) {
                f << "\\\\" << line << "\n";
            }
        }
        if (b.type == BlockType::Fuzzy && b.fuzzy) {
            f << b.fuzzy->size() << " Count of Values\n";
            for (size_t i = 0; i < b.fuzzy->size(); i++) {
                f << b.fuzzy->x_at(i) << " " << b.fuzzy->membership_at(i) << " Index,Value\n";
            }
        }
    }

    static void load_block(std::ifstream& f, LogicBlock& b) {
        std::string line;
        std::getline(f, line); // blank
        std::getline(f, line); // ID line
        int type_int = 0;
        std::sscanf(line.c_str(), "ID %d Type %d", &b.id, &type_int);
        b.type = (BlockType)type_int;

        std::getline(f, line); // Packed
        int p = 0;
        std::sscanf(line.c_str(), "Packed %d PackageID %d", &p, &b.pack_id);
        b.packed = (p == 1);

        std::getline(f, line); // x y
        std::sscanf(line.c_str(), "%d %d", &b.x, &b.y);

        if (b.type == BlockType::Fuzzy) {
            std::getline(f, line);
            std::sscanf(line.c_str(), "%d %d", &b.imin, &b.imax);
            std::getline(f, line);
            std::sscanf(line.c_str(), "%d %d", &b.omin, &b.omax);
        } else if (b.type == BlockType::In) {
            std::getline(f, line);
            std::sscanf(line.c_str(), "%d %d", &b.imin, &b.imax);
        }

        // Name
        std::getline(f, line);
        if (line.size() > 6) b.name = line.substr(6); // "Name: "

        // Description
        std::getline(f, line);
        if (line.size() > 13) b.description = line.substr(13); // "Description: "

        // Function text for In blocks
        if (b.type == BlockType::In) {
            std::getline(f, line);
            int count = 0;
            std::sscanf(line.c_str(), "%d", &count);
            b.function_text.clear();
            for (int i = 0; i < count; i++) {
                std::getline(f, line);
                if (line.size() >= 2 && line[0] == '\\' && line[1] == '\\')
                    line = line.substr(2);
                b.function_text += line + "\n";
            }
        }

        // Fuzzy points
        if (b.type == BlockType::Fuzzy) {
            std::getline(f, line);
            int count = 0;
            std::sscanf(line.c_str(), "%d", &count);
            b.fuzzy = crystal::FuzzySet<float>();
            for (int i = 0; i < count; i++) {
                std::getline(f, line);
                float ind = 0, val = 0;
                std::sscanf(line.c_str(), "%f %f", &ind, &val);
                b.fuzzy->add_point(ind, val);
            }
        }
    }
};

// ── FuzzyChartWidget ────────────────────────────────────────────────────────

class FuzzyChartWidget : public QWidget {
    Q_OBJECT
public:
    crystal::FuzzySet<float>* current_fuzzy = nullptr;
    int imin = 0, imax = 255, omin = 0, omax = 255;

    explicit FuzzyChartWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(200, 150);
        setMouseTracking(true);
    }

signals:
    void fuzzy_changed();

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), Qt::white);

        int margin = 30;
        QRect chart(margin, 10, width() - margin - 10, height() - margin - 10);
        p.setPen(Qt::lightGray);
        p.drawRect(chart);

        // Axes labels
        p.setPen(Qt::black);
        p.setFont(QFont("monospace", 8));
        p.drawText(margin, height() - 5, QString::number(imin));
        p.drawText(chart.right() - 20, height() - 5, QString::number(imax));
        p.drawText(2, chart.bottom(), "0");
        p.drawText(2, chart.top() + 10, "1");

        if (!current_fuzzy || current_fuzzy->empty()) return;

        // Draw membership function
        p.setPen(QPen(QColor(80, 180, 200), 2));
        auto& pts = current_fuzzy->points();
        auto to_screen = [&](float x, float m) -> QPointF {
            float fx = (x - imin) / float(imax - imin);
            float fy = 1.0f - m;
            return QPointF(chart.left() + fx * chart.width(),
                          chart.top() + fy * chart.height());
        };

        for (size_t i = 0; i + 1 < pts.size(); i++) {
            QPointF a = to_screen(pts[i].x, pts[i].membership);
            QPointF b = to_screen(pts[i + 1].x, pts[i + 1].membership);
            p.drawLine(a, b);
        }

        // Draw points
        p.setPen(Qt::black);
        p.setBrush(QColor(80, 180, 200));
        for (auto& pt : pts) {
            QPointF sp = to_screen(pt.x, pt.membership);
            p.drawEllipse(sp, 4, 4);
        }
    }

    void mousePressEvent(QMouseEvent* e) override {
        if (!current_fuzzy) return;
        int margin = 30;
        QRect chart(margin, 10, width() - margin - 10, height() - margin - 10);
        if (!chart.contains(e->pos())) return;

        float fx = float(e->pos().x() - chart.left()) / chart.width();
        float fy = 1.0f - float(e->pos().y() - chart.top()) / chart.height();
        fx = std::clamp(fx, 0.0f, 1.0f);
        fy = std::clamp(fy, 0.0f, 1.0f);

        float x = imin + fx * (imax - imin);
        float m = fy;

        if (e->button() == Qt::LeftButton) {
            current_fuzzy->add_point(x, m);
        } else if (e->button() == Qt::RightButton) {
            // Remove nearest point
            auto& pts = current_fuzzy->points();
            if (pts.empty()) return;
            float best_dist = 1e9f;
            float best_x = 0;
            for (auto& pt : pts) {
                float d = std::abs(pt.x - x) + std::abs(pt.membership - m);
                if (d < best_dist) { best_dist = d; best_x = pt.x; }
            }
            // Rebuild without that point
            crystal::FuzzySet<float> new_fs;
            for (auto& pt : pts) {
                if (pt.x != best_x) new_fs.add_point(pt.x, pt.membership);
            }
            *current_fuzzy = new_fs;
        }
        emit fuzzy_changed();
        update();
    }
};

// ── BlockItem ───────────────────────────────────────────────────────────────

static constexpr int BLOCK_W = 80;
static constexpr int BLOCK_H = 80;

class BlockItem : public QGraphicsItem {
public:
    // Copy block data so we don't hold a pointer into a vector
    int block_id;
    BlockType block_type;
    std::string block_name;
    bool block_selected;
    float block_value;

    // Pointer back to graph for position updates (graph itself is stable)
    LogicGraph* graph;

    BlockItem(LogicGraph* g, const LogicBlock& b) : graph(g) {
        block_id = b.id;
        block_type = b.type;
        block_name = b.name;
        block_selected = b.selected;
        block_value = (b.type == BlockType::In) ? b.in_value : b.out_value;
        setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemSendsGeometryChanges);
        setPos(b.x, b.y);
    }

    QRectF boundingRect() const override {
        return QRectF(-6, -4, BLOCK_W + 12, BLOCK_H + 8);
    }

    void paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) override {
        p->setRenderHint(QPainter::Antialiasing);

        QColor bg = block_color(block_type);
        QColor border = bg.darker(150);
        if (block_selected) {
            bg = bg.lighter(120);
            border = QColor(255, 220, 50);
        }

        // Drop shadow
        p->setPen(Qt::NoPen);
        p->setBrush(QColor(0, 0, 0, 40));
        p->drawRoundedRect(QRectF(3, 3, BLOCK_W, BLOCK_H), 8, 8);

        // Main body
        p->setBrush(bg);
        p->setPen(QPen(border, block_selected ? 2.5 : 1.5));
        p->drawRoundedRect(QRectF(0, 0, BLOCK_W, BLOCK_H), 8, 8);

        // Icon area (top half)
        draw_icon(p, block_type);

        // Type badge
        p->setPen(Qt::NoPen);
        p->setBrush(QColor(0, 0, 0, 80));
        p->drawRoundedRect(QRectF(4, 4, 28, 14), 4, 4);
        p->setPen(Qt::white);
        p->setFont(QFont("sans-serif", 7, QFont::Bold));
        p->drawText(QRectF(4, 4, 28, 14), Qt::AlignCenter, block_type_name(block_type));

        // Name (bottom area)
        p->setPen(Qt::white);
        p->setFont(QFont("sans-serif", 8, QFont::Bold));
        QString name = QString::fromStdString(block_name);
        QFontMetrics fm(p->font());
        if (fm.horizontalAdvance(name) > BLOCK_W - 8)
            name = fm.elidedText(name, Qt::ElideRight, BLOCK_W - 8);
        p->drawText(QRectF(0, BLOCK_H - 28, BLOCK_W, 14), Qt::AlignCenter, name);

        // Value display
        if (block_value >= 0) {
            p->setFont(QFont("monospace", 7));
            p->setPen(QColor(255, 255, 255, 200));
            p->drawText(QRectF(0, BLOCK_H - 16, BLOCK_W, 14), Qt::AlignCenter,
                        QString::number(block_value, 'f', 2));
        }

        // Connection ports
        // Input port (left side) — except for In blocks which have no input
        if (block_type != BlockType::In) {
            p->setPen(QPen(Qt::white, 1.5));
            p->setBrush(QColor(60, 60, 60));
            p->drawEllipse(QPointF(-4, BLOCK_H / 2), 5, 5);
            // Arrow-in indicator
            p->setPen(QPen(Qt::white, 1.5));
            p->drawLine(QPointF(-6, BLOCK_H / 2), QPointF(-2, BLOCK_H / 2));
        }

        // Output port (right side) — except for Out blocks which have no output
        if (block_type != BlockType::Out) {
            p->setPen(QPen(Qt::white, 1.5));
            p->setBrush(QColor(60, 60, 60));
            p->drawEllipse(QPointF(BLOCK_W + 4, BLOCK_H / 2), 5, 5);
            // Arrow-out indicator
            p->setPen(QPen(Qt::white, 1.5));
            p->drawLine(QPointF(BLOCK_W + 2, BLOCK_H / 2), QPointF(BLOCK_W + 6, BLOCK_H / 2));
        }

        // Selection highlight glow
        if (block_selected) {
            p->setPen(QPen(QColor(255, 220, 50, 100), 4));
            p->setBrush(Qt::NoBrush);
            p->drawRoundedRect(QRectF(-3, -3, BLOCK_W + 6, BLOCK_H + 6), 10, 10);
        }
    }

    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override {
        if (change == ItemPositionChange && graph) {
            QPointF newPos = value.toPointF();
            if (auto* b = graph->get(block_id)) {
                b->x = (int)newPos.x();
                b->y = (int)newPos.y();
            }
        }
        return QGraphicsItem::itemChange(change, value);
    }

private:
    void draw_icon(QPainter* p, BlockType type) const {
        p->save();
        QRectF icon_area(10, 16, BLOCK_W - 20, 30);
        p->setPen(QPen(Qt::white, 2));
        p->setBrush(Qt::NoBrush);

        switch (type) {
            case BlockType::In: {
                // Arrow pointing right
                float cx = icon_area.center().x(), cy = icon_area.center().y();
                float hw = 16, hh = 10;
                QPolygonF arrow;
                arrow << QPointF(cx - hw, cy - hh/2)
                      << QPointF(cx + hw/2, cy - hh/2)
                      << QPointF(cx + hw/2, cy - hh)
                      << QPointF(cx + hw, cy)
                      << QPointF(cx + hw/2, cy + hh)
                      << QPointF(cx + hw/2, cy + hh/2)
                      << QPointF(cx - hw, cy + hh/2);
                p->setBrush(QColor(255, 255, 255, 60));
                p->drawPolygon(arrow);
                break;
            }
            case BlockType::Out: {
                // Target/bullseye
                float cx = icon_area.center().x(), cy = icon_area.center().y();
                p->drawEllipse(QPointF(cx, cy), 12, 12);
                p->drawEllipse(QPointF(cx, cy), 7, 7);
                p->setBrush(Qt::white);
                p->drawEllipse(QPointF(cx, cy), 3, 3);
                break;
            }
            case BlockType::Or: {
                // OR gate shape
                float cx = icon_area.center().x(), cy = icon_area.center().y();
                p->setFont(QFont("sans-serif", 16, QFont::Bold));
                p->drawText(QRectF(cx - 12, cy - 12, 24, 24), Qt::AlignCenter, "+");
                break;
            }
            case BlockType::Xor: {
                // XOR symbol
                float cx = icon_area.center().x(), cy = icon_area.center().y();
                p->setFont(QFont("sans-serif", 16, QFont::Bold));
                p->drawText(QRectF(cx - 12, cy - 12, 24, 24), Qt::AlignCenter, "\u2295");
                break;
            }
            case BlockType::And: {
                // AND gate — ampersand
                float cx = icon_area.center().x(), cy = icon_area.center().y();
                p->setFont(QFont("sans-serif", 18, QFont::Bold));
                p->drawText(QRectF(cx - 12, cy - 12, 24, 24), Qt::AlignCenter, "&");
                break;
            }
            case BlockType::Fuzzy: {
                // Fuzzy membership curve (S-shape)
                QPainterPath path;
                float x0 = icon_area.left(), x1 = icon_area.right();
                float y0 = icon_area.bottom(), y1 = icon_area.top();
                path.moveTo(x0, y0);
                float steps = 20;
                for (int i = 1; i <= (int)steps; i++) {
                    float t = i / steps;
                    float x = x0 + t * (x1 - x0);
                    // S-curve: sigmoid-like
                    float s = 1.0f / (1.0f + std::exp(-10.0f * (t - 0.5f)));
                    float y = y0 + s * (y1 - y0);
                    path.lineTo(x, y);
                }
                p->drawPath(path);
                break;
            }
            case BlockType::Package: {
                // Package icon — box
                float cx = icon_area.center().x(), cy = icon_area.center().y();
                p->drawRect(QRectF(cx - 10, cy - 8, 20, 16));
                p->drawLine(QPointF(cx - 10, cy - 2), QPointF(cx + 10, cy - 2));
                break;
            }
        }
        p->restore();
    }
};

// ── AddBlockDialog ──────────────────────────────────────────────────────────

class AddBlockDialog : public QDialog {
    Q_OBJECT
public:
    QComboBox* type_combo;
    QLineEdit* name_edit;
    QLineEdit* desc_edit;
    QSpinBox* imin_spin;
    QSpinBox* imax_spin;
    QSpinBox* omin_spin;
    QSpinBox* omax_spin;
    QTextEdit* code_edit;
    QTextEdit* global_edit;

    // type_counts[i] = how many blocks of type i already exist (for auto-naming)
    int type_counts[7] = {};
    bool name_manually_edited = false;

    AddBlockDialog(const std::string& global, const LogicGraph& graph, QWidget* parent = nullptr)
        : QDialog(parent) {
        setWindowTitle("Add Block");
        setMinimumWidth(400);

        // Count existing blocks per type for auto-naming
        for (auto& b : graph.blocks)
            type_counts[(int)b.type]++;

        auto* layout = new QVBoxLayout(this);

        auto* form = new QFormLayout;
        type_combo = new QComboBox;
        type_combo->addItems({"In", "Out", "Or", "Xor", "And", "Fuzzy"});
        form->addRow("Type:", type_combo);

        name_edit = new QLineEdit;
        form->addRow("Name:", name_edit);

        desc_edit = new QLineEdit;
        form->addRow("Description:", desc_edit);

        imin_spin = new QSpinBox; imin_spin->setRange(-10000, 10000); imin_spin->setValue(0);
        imax_spin = new QSpinBox; imax_spin->setRange(-10000, 10000); imax_spin->setValue(255);
        omin_spin = new QSpinBox; omin_spin->setRange(-10000, 10000); omin_spin->setValue(0);
        omax_spin = new QSpinBox; omax_spin->setRange(-10000, 10000); omax_spin->setValue(255);
        form->addRow("Input Min:", imin_spin);
        form->addRow("Input Max:", imax_spin);
        form->addRow("Output Min:", omin_spin);
        form->addRow("Output Max:", omax_spin);
        layout->addLayout(form);

        code_edit = new QTextEdit;
        code_edit->setPlainText("return 0;");
        code_edit->setMaximumHeight(80);
        layout->addWidget(new QLabel("Function Code (In blocks):"));
        layout->addWidget(code_edit);

        global_edit = new QTextEdit;
        global_edit->setPlainText(QString::fromStdString(global));
        global_edit->setMaximumHeight(80);
        layout->addWidget(new QLabel("Global Code:"));
        layout->addWidget(global_edit);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        layout->addWidget(buttons);

        // Auto-name when type changes (unless user has manually edited the name)
        connect(name_edit, &QLineEdit::textEdited, this, [this]() {
            name_manually_edited = true;
        });
        connect(type_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
            bool is_fuzzy = (idx == 5);
            bool is_in = (idx == 0);
            omin_spin->setEnabled(is_fuzzy);
            omax_spin->setEnabled(is_fuzzy);
            code_edit->setEnabled(is_in);
            if (!name_manually_edited)
                update_auto_name(idx);
        });
        type_combo->setCurrentIndex(0);
        update_auto_name(0);
    }

    void update_auto_name(int type_idx) {
        static const char* names[] = {"In", "Out", "Or", "Xor", "And", "Fuzzy"};
        if (type_idx >= 0 && type_idx < 6) {
            int num = type_counts[type_idx] + 1;
            name_edit->setText(QString("%1%2").arg(names[type_idx]).arg(num));
        }
    }
};

// ── FuzzyBuilderWindow ─────────────────────────────────────────────────────

class FuzzyBuilderWindow : public QMainWindow {
    Q_OBJECT
public:
    LogicGraph graph;
    QGraphicsScene* scene;
    QGraphicsView* view;
    FuzzyChartWidget* chart;
    QComboBox* block_combo;
    QLabel* status_label;
    QString current_file;

    // Simulation panel
    QGroupBox* sim_group;
    QVBoxLayout* sim_layout;
    std::vector<std::pair<int, QDoubleSpinBox*>> sim_inputs; // block_id -> spinbox

    // Connection drawing state
    int connect_from_id = -1;
    QGraphicsLineItem* rubber_line = nullptr;

    // Block drag state
    int drag_block_id = -1;
    int drag_offset_x = 0, drag_offset_y = 0;

    // Pan state
    bool pan_active = false;
    QPoint pan_last;

    FuzzyBuilderWindow() {
        setWindowTitle("Crystal FuzzyBuilder");
        resize(1000, 700);

        // Menu bar
        auto* file_menu = menuBar()->addMenu("&File");
        file_menu->addAction("&New", QKeySequence::New, this, &FuzzyBuilderWindow::new_file);
        file_menu->addAction("&Open...", QKeySequence::Open, this, &FuzzyBuilderWindow::open_file);
        file_menu->addAction("&Save", QKeySequence::Save, this, &FuzzyBuilderWindow::save_file);
        file_menu->addAction("Save &As...", this, &FuzzyBuilderWindow::save_file_as);
        file_menu->addSeparator();
        file_menu->addAction("Export &C Header...", this, &FuzzyBuilderWindow::export_header);
        file_menu->addSeparator();
        file_menu->addAction("&Quit", QKeySequence::Quit, this, &QWidget::close);

        auto* edit_menu = menuBar()->addMenu("&Edit");
        edit_menu->addAction("&Delete Selected", QKeySequence::Delete, this, &FuzzyBuilderWindow::delete_selected);
        edit_menu->addAction("&Evaluate Logic", this, &FuzzyBuilderWindow::evaluate_logic);

        // Toolbar
        auto* toolbar = addToolBar("Main");
        toolbar->addAction("Add Block", this, &FuzzyBuilderWindow::add_block);
        toolbar->addAction("Evaluate", this, &FuzzyBuilderWindow::evaluate_logic);

        // Central splitter
        auto* splitter = new QSplitter(Qt::Horizontal);

        // Left panel
        auto* left = new QWidget;
        auto* left_layout = new QVBoxLayout(left);

        block_combo = new QComboBox;
        left_layout->addWidget(new QLabel("In/Fuzzy Blocks:"));
        left_layout->addWidget(block_combo);

        auto* add_btn = new QPushButton("Add Block...");
        connect(add_btn, &QPushButton::clicked, this, &FuzzyBuilderWindow::add_block);
        left_layout->addWidget(add_btn);

        auto* clear_btn = new QPushButton("Clear Fuzzy Points");
        connect(clear_btn, &QPushButton::clicked, this, &FuzzyBuilderWindow::clear_fuzzy);
        left_layout->addWidget(clear_btn);

        auto* opt_btn = new QPushButton("Optimize Fuzzy");
        connect(opt_btn, &QPushButton::clicked, this, &FuzzyBuilderWindow::optimize_fuzzy);
        left_layout->addWidget(opt_btn);

        auto* inc_btn = new QPushButton("Increase Samples");
        connect(inc_btn, &QPushButton::clicked, this, &FuzzyBuilderWindow::increase_samples);
        left_layout->addWidget(inc_btn);

        chart = new FuzzyChartWidget;
        connect(chart, &FuzzyChartWidget::fuzzy_changed, this, &FuzzyBuilderWindow::on_fuzzy_changed);
        left_layout->addWidget(chart);

        // Simulation panel
        sim_group = new QGroupBox("Simulate");
        sim_layout = new QVBoxLayout(sim_group);
        auto* sim_run_btn = new QPushButton("Run Simulation");
        connect(sim_run_btn, &QPushButton::clicked, this, &FuzzyBuilderWindow::run_simulation);
        sim_layout->addWidget(sim_run_btn);
        left_layout->addWidget(sim_group);
        left_layout->addStretch();

        connect(block_combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &FuzzyBuilderWindow::on_combo_changed);

        splitter->addWidget(left);

        // Graphics view
        scene = new QGraphicsScene(this);
        scene->setSceneRect(0, 0, 2000, 2000);
        scene->setBackgroundBrush(QColor(240, 242, 245));
        view = new QGraphicsView(scene);
        view->setRenderHint(QPainter::Antialiasing);
        view->setDragMode(QGraphicsView::NoDrag);
        view->setMouseTracking(true);
        view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        view->viewport()->installEventFilter(this);
        splitter->addWidget(view);

        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 3);
        setCentralWidget(splitter);

        status_label = new QLabel("Ready");
        statusBar()->addWidget(status_label);
    }

protected:
    bool eventFilter(QObject* obj, QEvent* event) override {
        if (obj != view->viewport())
            return QMainWindow::eventFilter(obj, event);

        if (event->type() == QEvent::MouseButtonPress) {
            auto* me = static_cast<QMouseEvent*>(event);
            QPointF sp = view->mapToScene(me->pos());
            if (me->button() == Qt::LeftButton) {
                handle_left_click(sp, me->modifiers());
                return true; // consume so items don't get it
            }
            if (me->button() == Qt::MiddleButton) {
                // Start panning
                pan_active = true;
                pan_last = me->pos();
                view->setCursor(Qt::ClosedHandCursor);
                return true;
            }
        }
        if (event->type() == QEvent::MouseButtonRelease) {
            auto* me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::MiddleButton && pan_active) {
                pan_active = false;
                view->setCursor(Qt::ArrowCursor);
                return true;
            }
        }
        if (event->type() == QEvent::MouseMove) {
            auto* me = static_cast<QMouseEvent*>(event);
            QPointF sp = view->mapToScene(me->pos());

            // Panning with middle mouse button
            if (pan_active) {
                QPoint delta = me->pos() - pan_last;
                pan_last = me->pos();
                view->horizontalScrollBar()->setValue(view->horizontalScrollBar()->value() - delta.x());
                view->verticalScrollBar()->setValue(view->verticalScrollBar()->value() - delta.y());
                return true;
            }

            // Rubber band line for connection-in-progress
            if (rubber_line) {
                auto* from = graph.get(connect_from_id);
                if (from) {
                    rubber_line->setLine(from->x + BLOCK_W + 4,
                                         from->y + BLOCK_H / 2,
                                         sp.x(), sp.y());
                }
                return true;
            }

            // Dragging a block
            if (drag_block_id >= 0) {
                auto* b = graph.get(drag_block_id);
                if (b) {
                    b->x = (int)sp.x() - drag_offset_x;
                    b->y = (int)sp.y() - drag_offset_y;
                    refresh_scene();
                }
                return true;
            }
        }
        if (event->type() == QEvent::MouseButtonRelease) {
            auto* me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::LeftButton && drag_block_id >= 0) {
                drag_block_id = -1;
                return true;
            }
        }
        if (event->type() == QEvent::Wheel) {
            auto* we = static_cast<QWheelEvent*>(event);
            double factor = (we->angleDelta().y() > 0) ? 1.15 : 1.0 / 1.15;
            view->scale(factor, factor);
            return true;
        }
        return QMainWindow::eventFilter(obj, event);
    }

private slots:
    void new_file() {
        graph.clear();
        current_file.clear();
        refresh();
    }

    void open_file() {
        QString path = QFileDialog::getOpenFileName(this, "Open FuzzyBuilder File", {},
            "FuzzyBuilder Files (*.h *.fb);;All Files (*)");
        if (path.isEmpty()) return;
        graph.load(path.toStdString());
        current_file = path;
        refresh();
    }

    void save_file() {
        if (current_file.isEmpty()) { save_file_as(); return; }
        graph.save(current_file.toStdString());
        status_label->setText("Saved: " + current_file);
    }

    void save_file_as() {
        QString path = QFileDialog::getSaveFileName(this, "Save FuzzyBuilder File", {},
            "FuzzyBuilder Files (*.h);;All Files (*)");
        if (path.isEmpty()) return;
        current_file = path;
        save_file();
    }

    void export_header() {
        QString path = QFileDialog::getSaveFileName(this, "Export C Header", "fuzzy_logic.h",
            "C Headers (*.h);;All Files (*)");
        if (path.isEmpty()) return;
        std::ofstream f(path.toStdString());
        graph.export_c_header(f);
        status_label->setText("Exported: " + path);
    }

    void add_block() {
        AddBlockDialog dlg(graph.global_code, graph, this);
        if (dlg.exec() != QDialog::Accepted) return;

        LogicBlock b;
        b.id = graph.unique_id();
        b.type = (BlockType)dlg.type_combo->currentIndex();
        b.name = dlg.name_edit->text().toStdString();
        b.description = dlg.desc_edit->text().toStdString();
        b.function_text = dlg.code_edit->toPlainText().toStdString();
        if (b.function_text.empty() || b.function_text.back() != '\n')
            b.function_text += '\n';
        b.imin = dlg.imin_spin->value();
        b.imax = dlg.imax_spin->value();
        b.omin = dlg.omin_spin->value();
        b.omax = dlg.omax_spin->value();

        // Position based on type
        int count = graph.count(b.type);
        switch (b.type) {
            case BlockType::In:    b.x = 50; break;
            case BlockType::Out:   b.x = 800; break;
            case BlockType::Fuzzy: b.x = 200; break;
            case BlockType::Or:    b.x = 400; break;
            case BlockType::And:   b.x = 500; break;
            case BlockType::Xor:   b.x = 350; break;
            default:               b.x = 300; break;
        }
        b.y = 100 * count + 50;

        if (b.type == BlockType::Fuzzy) {
            b.fuzzy = crystal::FuzzySet<float>();
        }

        graph.global_code = dlg.global_edit->toPlainText().toStdString();
        if (graph.global_code.empty() || graph.global_code.back() != '\n')
            graph.global_code += '\n';

        int bx = b.x, by = b.y;
        graph.blocks.push_back(std::move(b));
        refresh();
        view->ensureVisible(bx, by, BLOCK_W, BLOCK_H, 50, 50);
    }

    void delete_selected() {
        std::vector<int> to_delete;
        for (auto& b : graph.blocks) {
            if (b.selected) to_delete.push_back(b.id);
        }
        for (int id : to_delete) graph.remove_block(id);
        refresh();
    }

    void evaluate_logic() {
        graph.update_logic();
        refresh();
        status_label->setText("Logic evaluated");
    }

    void run_simulation() {
        // Set In block values from the spinboxes
        for (auto& [bid, spin] : sim_inputs) {
            auto* b = graph.get(bid);
            if (b) b->in_value = (float)spin->value();
        }
        // Evaluate
        graph.update_logic();
        refresh_scene();

        // Show results in status bar
        QString results;
        for (auto& b : graph.blocks) {
            if (b.type == BlockType::Out) {
                if (!results.isEmpty()) results += "  |  ";
                results += QString("%1 = %2")
                    .arg(QString::fromStdString(b.name))
                    .arg(b.out_value, 0, 'f', 3);
            }
        }
        status_label->setText(results.isEmpty() ? "No output blocks" : "Results: " + results);
    }

    void clear_fuzzy() {
        int idx = block_combo->currentIndex();
        if (idx < 0) return;
        int id = block_combo->currentData().toInt();
        auto* b = graph.get(id);
        if (b && b->fuzzy) {
            b->fuzzy->clear();
            chart->update();
        }
    }

    void optimize_fuzzy() {
        int idx = block_combo->currentIndex();
        if (idx < 0) return;
        int id = block_combo->currentData().toInt();
        auto* b = graph.get(id);
        if (b && b->fuzzy) {
            b->fuzzy->optimize();
            chart->update();
        }
    }

    void increase_samples() {
        int idx = block_combo->currentIndex();
        if (idx < 0) return;
        int id = block_combo->currentData().toInt();
        auto* b = graph.get(id);
        if (b && b->fuzzy) {
            b->fuzzy->increase_samples();
            chart->update();
        }
    }

    void on_combo_changed(int idx) {
        if (idx < 0) { chart->current_fuzzy = nullptr; chart->update(); return; }
        int id = block_combo->currentData().toInt();
        auto* b = graph.get(id);
        if (b && b->type == BlockType::Fuzzy && b->fuzzy) {
            chart->current_fuzzy = &(*b->fuzzy);
            chart->imin = b->imin;
            chart->imax = b->imax;
        } else if (b && b->type == BlockType::In) {
            chart->current_fuzzy = nullptr;
            chart->imin = b->imin;
            chart->imax = b->imax;
        } else {
            chart->current_fuzzy = nullptr;
        }
        chart->update();
    }

    void on_fuzzy_changed() {
        refresh_scene();
    }

private:
    // Port hit radius
    static constexpr float PORT_RADIUS = 10.0f;

    bool hit_output_port(const LogicBlock& b, QPointF pos) const {
        float px = b.x + BLOCK_W + 4, py = b.y + BLOCK_H / 2;
        float dx = pos.x() - px, dy = pos.y() - py;
        return (dx * dx + dy * dy) <= PORT_RADIUS * PORT_RADIUS;
    }

    bool hit_input_port(const LogicBlock& b, QPointF pos) const {
        float px = b.x - 4, py = b.y + BLOCK_H / 2;
        float dx = pos.x() - px, dy = pos.y() - py;
        return (dx * dx + dy * dy) <= PORT_RADIUS * PORT_RADIUS;
    }

    bool hit_block_body(const LogicBlock& b, QPointF pos) const {
        return pos.x() >= b.x && pos.x() <= b.x + BLOCK_W &&
               pos.y() >= b.y && pos.y() <= b.y + BLOCK_H;
    }

    void handle_left_click(QPointF pos, Qt::KeyboardModifiers mods) {
        // ── If connection in progress, try to complete it ──
        if (connect_from_id >= 0) {
            // Look for an input port hit
            for (auto& b : graph.blocks) {
                if (b.id != connect_from_id && b.type != BlockType::In &&
                    (hit_input_port(b, pos) || hit_block_body(b, pos))) {
                    graph.connections.push_back({connect_from_id, b.id});
                    break;
                }
            }
            // End connection mode regardless
            if (rubber_line) { scene->removeItem(rubber_line); delete rubber_line; rubber_line = nullptr; }
            connect_from_id = -1;
            view->setCursor(Qt::ArrowCursor);
            refresh_scene();
            return;
        }

        // ── Check if clicking an output port to start a connection ──
        for (auto& b : graph.blocks) {
            if (b.type != BlockType::Out && hit_output_port(b, pos)) {
                connect_from_id = b.id;
                rubber_line = scene->addLine(
                    b.x + BLOCK_W + 4, b.y + BLOCK_H / 2,
                    pos.x(), pos.y(), QPen(QColor(80, 80, 80), 2, Qt::DashLine));
                rubber_line->setZValue(10);
                view->setCursor(Qt::CrossCursor);
                status_label->setText(QString("Connecting from %1 — click target block to complete, or click empty space to cancel")
                    .arg(QString::fromStdString(b.name)));
                return;
            }
        }

        // ── Check if clicking a block body to select or start drag ──
        for (auto& b : graph.blocks) {
            if (hit_block_body(b, pos)) {
                if (!(mods & Qt::ShiftModifier)) graph.unselect_all();
                b.selected = !b.selected;
                // Start drag
                drag_block_id = b.id;
                drag_offset_x = (int)pos.x() - b.x;
                drag_offset_y = (int)pos.y() - b.y;
                refresh_scene();
                return;
            }
        }

        // ── Clicked empty space: deselect all ──
        graph.unselect_all();
        refresh_scene();
    }

    void refresh() {
        update_combo();
        rebuild_sim_panel();
        refresh_scene();
    }

    void update_combo() {
        block_combo->blockSignals(true);
        block_combo->clear();
        for (auto& b : graph.blocks) {
            if (b.type == BlockType::In || b.type == BlockType::Fuzzy) {
                QString label = QString("%1:%2").arg(block_type_name(b.type),
                                                      QString::fromStdString(b.name));
                block_combo->addItem(label, b.id);
            }
        }
        block_combo->blockSignals(false);
        if (block_combo->count() > 0) {
            block_combo->setCurrentIndex(0);
            on_combo_changed(0);
        }
    }

    void rebuild_sim_panel() {
        // Remove old spinboxes (keep the Run button at index 0)
        sim_inputs.clear();
        // Remove all widgets except the button (which is at index 0)
        while (sim_layout->count() > 1) {
            auto* item = sim_layout->takeAt(1);
            if (item->widget()) { delete item->widget(); }
            delete item;
        }

        // Add a labeled spinbox for each In block
        for (auto& b : graph.blocks) {
            if (b.type != BlockType::In) continue;
            auto* row = new QWidget;
            auto* hl = new QHBoxLayout(row);
            hl->setContentsMargins(0, 0, 0, 0);
            auto* label = new QLabel(QString::fromStdString(b.name));
            label->setMinimumWidth(60);
            hl->addWidget(label);
            auto* spin = new QDoubleSpinBox;
            spin->setRange(b.imin, b.imax);
            spin->setValue(b.in_value >= 0 ? b.in_value : 0);
            spin->setSingleStep((b.imax - b.imin) / 100.0);
            spin->setDecimals(2);
            hl->addWidget(spin);
            sim_layout->addWidget(row);
            sim_inputs.push_back({b.id, spin});
        }
    }

    void refresh_scene() {
        // Save rubber line state
        bool had_rubber = (rubber_line != nullptr);
        QLineF rubber_geo;
        if (had_rubber) rubber_geo = rubber_line->line();

        scene->clear();
        rubber_line = nullptr;

        // Draw connections as curved arrows
        for (auto& c : graph.connections) {
            auto* from = graph.get(c.from_id);
            auto* to = graph.get(c.to_id);
            if (from && to) {
                QPointF p1(from->x + BLOCK_W + 4, from->y + BLOCK_H / 2);
                QPointF p2(to->x - 4, to->y + BLOCK_H / 2);
                // Draw with arrowhead
                auto* line = scene->addLine(QLineF(p1, p2), QPen(QColor(80, 80, 80), 2));
                line->setZValue(-1);
                // Arrowhead
                double angle = std::atan2(p2.y() - p1.y(), p2.x() - p1.x());
                double alen = 8;
                QPointF a1(p2.x() - alen * std::cos(angle - 0.4),
                           p2.y() - alen * std::sin(angle - 0.4));
                QPointF a2(p2.x() - alen * std::cos(angle + 0.4),
                           p2.y() - alen * std::sin(angle + 0.4));
                QPolygonF arrowhead;
                arrowhead << p2 << a1 << a2;
                auto* ah = scene->addPolygon(arrowhead, QPen(Qt::NoPen), QBrush(QColor(80, 80, 80)));
                ah->setZValue(-1);
            }
        }

        // Draw blocks
        for (auto& b : graph.blocks) {
            auto* item = new BlockItem(&graph, b);
            scene->addItem(item);
        }

        // Restore rubber line
        if (had_rubber) {
            rubber_line = scene->addLine(rubber_geo, QPen(Qt::black, 1, Qt::DashLine));
        }

        status_label->setText(QString("Blocks: %1  Connections: %2")
            .arg(graph.blocks.size()).arg(graph.connections.size()));
    }
};

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("Crystal FuzzyBuilder");

    FuzzyBuilderWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
