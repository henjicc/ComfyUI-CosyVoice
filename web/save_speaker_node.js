import { app } from "../../../scripts/app.js";

// 显示保存说话人信息的节点
app.registerExtension({
    name: "ComfyUI-CosyVoice.SaveSpeakerNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CosyVoice2SaveSpeaker") {
            // 添加多行文本显示框
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                if (message.text) {
                    // 创建显示区域
                    let widget = this.widgets.find(w => w.name === "save_info");
                    if (!widget) {
                        // 创建新的文本显示widget
                        const container = document.createElement("div");
                        const textBox = document.createElement("div");
                        container.appendChild(textBox);
                        
                        widget = this.addDOMWidget("save_info", "customtext", container, {
                            getValue() {
                                return textBox.textContent || "";
                            },
                            setValue(v) {
                                if (textBox) {
                                    textBox.textContent = v || "";
                                }
                            }
                        });
                        
                        // 设置样式
                        container.style.background = "#222";
                        container.style.color = "#31EC88";
                        container.style.padding = "10px";
                        container.style.borderRadius = "5px";
                        container.style.overflow = "auto";
                        container.style.maxHeight = "150px";
                        
                        textBox.style.whiteSpace = "pre-wrap";
                        textBox.style.fontSize = "12px";
                        textBox.style.wordBreak = "break-word";
                        
                        // 设置widget属性
                        widget.serializeValue = () => {
                            return textBox.textContent || "";
                        };
                    }
                    
                    // 安全地更新文本内容
                    if (widget.element && widget.element.firstChild) {
                        widget.element.firstChild.textContent = message.text[0];
                    } else if (widget.element) {
                        widget.element.textContent = message.text[0];
                    }
                    
                    // 触发节点重绘
                    this.setSize(this.computeSize());
                }
            };
        }
    }
});