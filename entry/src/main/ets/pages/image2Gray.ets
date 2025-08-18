import router from '@ohos.router';
import image from '@ohos.multimedia.image';
// import Logger from '../../../util/Logger';
import testNapi from 'libentry.so';

// import { TitleBar } from '../../../common/TitleBar'

const TAG: string = 'opencv_gray';

@Entry
@Component
struct Image2Gray {
  private pixelMapFormat: image.PixelMapFormat = 3;
  @State isGray: Boolean = false;
  @State imagePixelMap: PixelMap | undefined = undefined;

  build() {
    Column() {
      // 标题

      Column() {
        Column() {
          if (this.isGray == false) {
            Image($rawfile('lena.jpg'))
              .margin({ left: 24, right: 24 })
              .objectFit(ImageFit.Contain)
              .id('backBtn')
          } else {
            Image(this.imagePixelMap)
              .margin({ left: 24, right: 24 })
              .objectFit(ImageFit.Contain)
              .id('backBtn')
          }
        }
        .width('100%')
        .height(480)
        .alignItems(HorizontalAlign.Center)
        .justifyContent(FlexAlign.Start)

        Row() {
          if (this.isGray) {
            Button($r('app.string.image_gray'), { type: ButtonType.Capsule })
              .backgroundColor(Color.Gray)
              .width(120)
              .height(48)
              .fontSize(16)
              .fontWeight(500)
              .fontColor(Color.White)
              .margin({ left: 24 })
              .id('imageGray')
              .enabled(false)

          } else {
            Button($r('app.string.image_gray'), { type: ButtonType.Capsule })
              .backgroundColor(Color.Blue)
              .width(120)
              .height(48)
              .fontSize(16)
              .fontWeight(500)
              .fontColor(Color.White)
              .margin({ left: 24 })
              .id('imageGray')
              .onClick(() => {
                let pixelInfo: testNapi.PixelInfo =
                  testNapi.img2Gray((this.getUIContext().getHostContext() as Context).resourceManager, '', 'lena.jpg');
                console.info(TAG, `pixelInfo buffSize: ${pixelInfo.buffSize}`);
                const color = pixelInfo.byteBuffer;
                let bufferArr = new Uint8Array(color);
                console.info("--opencv raw arraybuffer--lenth:%d", bufferArr.length);
                console.info("--opencv raw arraybuffer--0---:%s", bufferArr);

                let opts: image.InitializationOptions = {
                  editable: true,
                  pixelFormat: this.pixelMapFormat,
                  size: { height: pixelInfo.rows, width: pixelInfo.cols }
                }
                image.createPixelMap(pixelInfo.byteBuffer, opts, (error, pixelmap) => {
                  if (error) {
                    console.error(TAG, `Failed to create pixelmap error_code ${error.code}`);
                  } else {
                    console.info(TAG, 'Succeeded in creating pixelmap.');
                    this.imagePixelMap = pixelmap;
                  }
                })
                this.isGray = true;
              })
          }
          Button($r('app.string.image_recover'), { type: ButtonType.Capsule })
            .backgroundColor(Color.Blue)
            .width(120)
            .height(48)
            .fontSize(16)
            .fontWeight(500)
            .fontColor(Color.White)
            .margin({ right: 24 })
            .id('imageRecover')
            .onClick(() => {
              this.isGray = false;
            })
        }
        .width('100%')
        .height(48)
        .alignItems(VerticalAlign.Center)
        .justifyContent(FlexAlign.SpaceBetween)
      }
      .width('100%')
    }
    .height('100%')
    .width('100%')
    .backgroundColor(Color.Gray)
  }
}